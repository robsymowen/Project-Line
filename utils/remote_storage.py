'''
    utilities for storing model outputs in remote s3 buckets
'''
import os
import boto3
from tqdm import tqdm
import posixpath
import botocore.exceptions
import hashlib
import json
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse

def listify(o):
    if o is None: return []
    if isinstance(o, list): return o
    if isinstance(o, str): return [o]
    if isinstance(o, Iterable): return list(o)
    return [o]

def setify(o): return o if isinstance(o, set) else set(listify(o))

def _get_files(p, fs, extensions=None, contains=None):
    p = Path(p)
    res = [p/f for f in fs if not f.startswith('.')
           and ((not extensions) or f'.{f.split(".")[-1].lower()}' in extensions)
           and ((not contains) or contains in f.lower())]
    return res
                
def get_files(path, extensions=None, recurse=None, include=None, contains=None, sort=True):
    path = Path(path)
    extensions = setify(extensions)
    extensions = {e.lower() for e in extensions}
    if recurse:
        res = []
        for p,d,f in os.walk(path): # returns (dirpath, dirnames, filenames)
            if include is not None: d[:] = [o for o in d if o in include]
            else:                   d[:] = [o for o in d if not o.startswith('.')]            
            res += _get_files(p, f, extensions, contains)
        return sorted(res) if sort else res
    else:
        f = [o.name for o in os.scandir(path) if o.is_file()]    
        res = _get_files(path, f, extensions, contains)
        return sorted(res) if sort else res

def has_hash(filename):
    return len(Path(filename).stem.split("-")) == 2

def get_file_hash(filename):
    with open(filename,"rb") as f:
        bytes = f.read() # read entire file as bytes
        readable_hash = hashlib.sha256(bytes).hexdigest();
        
    return readable_hash

def check_hashid(hashid, weights_path):
    weights_hash = get_file_hash(weights_path)
    assert weights_hash.startswith(hashid), f"Oops, expected weights_hash to start with {hashid}, got {weights_hash}"
    return True

def get_subfolder(weight_file, split_from="logs"):
    parts = weight_file.parts
    subfolder = f"{os.path.sep}".join(parts[parts.index(split_from):-1])
    
    return subfolder
    
def get_public_s3_object_url(bucket_name, object_name):
    s3_client = boto3.client('s3')
    response = s3_client.generate_presigned_url('get_object',
                                                Params={'Bucket': bucket_name,
                                                        'Key': object_name},
                                                ExpiresIn=0,
                                                HttpMethod='GET')

    return response

def get_url(bucket_name, object_name):

    # Create an S3 client
    s3_client = boto3.client('s3')

    # Get the bucket location
    bucket_location = s3_client.get_bucket_location(Bucket=bucket_name)['LocationConstraint']

    # Construct the object URL
    object_url = f"https://{bucket_name}.s3.{bucket_location}.amazonaws.com/{object_name}"
    
    return object_url

class RemoteStorage:
    def __init__(self, local_dir, bucket_name, bucket_subfolder, acl='public-read', 
                 profile='wasabi', endpoint_url='https://s3.wasabisys.com',
                 hash_length=10):
        self.local_dir = local_dir
        self.bucket_name = bucket_name
        self.bucket_subfolder = bucket_subfolder
        self.bucket_path = posixpath.join(bucket_name, bucket_subfolder)
        self.acl = acl
        self.profile = profile
        self.endpoint_url = endpoint_url
        self.hash_length = hash_length
        self.session = boto3.Session(profile_name=self.profile)
        self.s3 = self.session.resource('s3', endpoint_url=self.endpoint_url)
        self.bucket = self.s3.Bucket(bucket_name)

    def upload_file_to_s3(self, file_path, bucket_name, object_name, acl=None, verbose=False):
        object_name = object_name.replace("//", "/")
        acl = self.acl if acl is None else acl
        bucket = self.bucket        
        base_url = f"https://{bucket_name}.{urlparse(self.endpoint_url).netloc}"
        object_url = urljoin(base_url, object_name)
        
        try:
            s3_file_size = bucket.Object(object_name).content_length
            local_file_size = os.path.getsize(file_path)
            if s3_file_size == local_file_size:
                if verbose: 
                    print(f"The file '{object_name}' already exists in the S3 bucket '{bucket_name}' and has the same size. The file will not be re-uploaded.\n")
                    print(object_url+"\n")
                return
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                # The key does not exist.
                pass
            elif e.response['Error']['Code'] == 403:
                # Unauthorized, including invalid bucket
                raise e
            else:
              # Something else has gone wrong.
              raise e

        # Upload the file
        self.s3.Object(bucket_name, object_name).put(Body=open(file_path, 'rb'), ACL=acl)
        if verbose: 
            print(f"The file '{object_name}' has been uploaded to the S3 bucket '{bucket_name}'.\n")            
            print(object_url+"\n")
    
    def upload_files(self, files, bucket_name=None, progress=False, verbose=False):
        # check for bucket override
        bucket_name = self.bucket_name if bucket_name is None else bucket_name
        
        # Loop through local files, upload to bucket_subfolder
        file_progress = tqdm(files) if progress else files
        for local_file in file_progress:
            filename = Path(local_file).name
            object_name = posixpath.join(self.bucket_subfolder, filename).replace("//","/")
            self.upload_file_to_s3(local_file, bucket_name, object_name, verbose=verbose)
    
    def upload_logs(self, local_dir=None, extensions=['.txt'], progress=False, verbose=False):
        local_dir = self.local_dir if local_dir is None else local_dir
        files = get_files(local_dir, extensions=extensions, recurse=True)   
        self.upload_files(files, progress=progress, verbose=verbose)
    
    def init_logs(self, verbose=False):
        filenames = [
            os.path.join(self.local_dir, 'log_train.txt'), 
            os.path.join(self.local_dir, 'log_val.txt'),
            os.path.join(self.local_dir, 'log_test.txt')
        ]
        for filename in filenames:
            Path(filename).touch(exist_ok=True)
        self.upload_logs(verbose=verbose)
        
    def upload_final_results(self, local_dir=None, acl='public-read', extensions=[".pt", ".pth", ".pth.tar"]):
        local_dir = self.local_dir if local_dir is None else local_dir
        acl = self.acl if acl is None else acl
        
        # get a list of final_weights.pt files in local_dir
        files = get_files(local_dir, extensions=extensions, recurse=True, contains='final_weights.pt')

        # Loop through weights files; append weights hash_id to weights, params, logs; upload to s3
        for local_weights_filename in files:
            local_params_filename = next(iter(local_weights_filename.parent.glob("*params.json")))
            local_log_train_filename = next(iter(local_weights_filename.parent.glob("log*train*")))  
            local_log_val_filename = next(iter(local_weights_filename.parent.glob("log*val*"))) 
            local_log_test_filename = next(iter(local_weights_filename.parent.glob("log*test*"))) 
            
            assert 'local_params_filename' in locals() and local_params_filename.is_file(), "Missing params.json"
            assert 'local_log_train_filename' in locals() and local_log_train_filename.is_file(), "Missing log_train file"
            assert 'local_log_val_filename' in locals() and local_log_val_filename.is_file(), "Missing log_val file"
            assert 'local_log_test_filename' in locals() and local_log_test_filename.is_file(), "Missing log_test file"

            # compute hash id (content-specific id of this weights file, used by pytorch to verify weights)
            print(f"computing hash_id: {local_weights_filename}")
            hash_id = get_file_hash(local_weights_filename)[0:self.hash_length]  
            print(f"\nhash_id: {hash_id}")
            print("")
            
            # then load the params file to get the architecture name
            #with open(local_params_filename, "r") as f: 
            #    params = json.loads(f.read())
            #arch = params['model.arch']
            
            print(f'local_log_folder:\n{local_weights_filename.parent}')
            print(f'local_weights_filename:\t\t{Path(local_weights_filename).name}')
            print(f'local_params_filename:\t\t{Path(local_params_filename).name}')
            print(f'local_log_train_filename:\t{Path(local_log_train_filename).name}')
            print(f'local_log_val_filename:\t\t{Path(local_log_val_filename).name}')
            print(f'local_log_test_filename:\t\t{Path(local_log_test_filename).name}')
            print("")
            
            # define our bucket filenames
            weight_bucket_file = posixpath.join(self.bucket_subfolder, f'final_weights-{hash_id}.pth')
            params_bucket_file = posixpath.join(self.bucket_subfolder, f'params-{hash_id}.json')
            log_train_bucket_file = posixpath.join(self.bucket_subfolder, f'log_train-{hash_id}.txt')
            log_val_bucket_file = posixpath.join(self.bucket_subfolder, f'log_val-{hash_id}.txt')
            log_test_bucket_file = posixpath.join(self.bucket_subfolder, f'log_test-{hash_id}.txt')
            
            print(f'remote bucket_path:\n{self.bucket_path}')
            print(f'weight_bucket_file:\t{Path(weight_bucket_file).name}')
            print(f'params_bucket_file:\t{Path(params_bucket_file).name}')
            print(f'log_train_bucket_file:\t{Path(log_train_bucket_file).name}')
            print(f'log_val_bucket_file:\t{Path(log_val_bucket_file).name}')
            print(f'log_test_bucket_file:\t{Path(log_test_bucket_file).name}')
            print("")
            
            # upload files to s3
            self.upload_file_to_s3(local_weights_filename, self.bucket_name, weight_bucket_file, acl=acl, verbose=True)
            self.upload_file_to_s3(local_params_filename, self.bucket_name, params_bucket_file, acl=acl, verbose=True)
            self.upload_file_to_s3(local_log_train_filename, self.bucket_name, log_train_bucket_file, acl=acl, verbose=True)
            self.upload_file_to_s3(local_log_val_filename, self.bucket_name, log_val_bucket_file, acl=acl, verbose=True)