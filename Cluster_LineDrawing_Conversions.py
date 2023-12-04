{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "collapsed_sections": [
        "3iYq1vu5RO52",
        "x6h2ZLl8UNQQ",
        "uJTSk8Qgmccg"
      ],
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/robsymowen/Project-Line/blob/main/Cluster_LineDrawing_Conversions.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Input Directory\n",
        "IMAGENET_ROOT_DIR = \"/content/drive/MyDrive/Classes/Third/PSY2356R/ThomasGarity/LineDrawing/datasets/imagenette2-320\"\n",
        "\n",
        "# Output Directory\n",
        "OUTPUT_DIR = '/content/drive/MyDrive/Classes/Third/PSY2356R/ThomasGarity/LineDrawing'"
      ],
      "metadata": {
        "id": "mGx4UGDt_2-U"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **Load Dataset**\n",
        "- Director to dataset is stored under 'IMAGENET_ROOT_DIR'"
      ],
      "metadata": {
        "id": "3iYq1vu5RO52"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import argparse\n",
        "from glob import glob\n",
        "import os\n",
        "import subprocess\n",
        "import shutil\n",
        "\n",
        "# Create the parser\n",
        "parser = argparse.ArgumentParser(description='Process line drawing conversion arguments.')\n",
        "\n",
        "# Add arguments\n",
        "parser.add_argument('imagenet_root_dir', type=str, help='The root directory of the ImageNet dataset')\n",
        "parser.add_argument('output_dir', type=str, help='The directory where output will be stored')\n",
        "\n",
        "# Parse the arguments\n",
        "args = parser.parse_args()\n",
        "\n",
        "# Use the parsed arguments\n",
        "IMAGENET_ROOT_DIR = args.imagenet_root_dir\n",
        "OUTPUT_DIR = args.output_dir"
      ],
      "metadata": {
        "id": "oyFG0ZWvSjpd",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 194
        },
        "outputId": "afb5e802-76c8-44dd-898c-fa9f2f26e855"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "usage: colab_kernel_launcher.py [-h] imagenet_root_dir output_dir\n",
            "colab_kernel_launcher.py: error: the following arguments are required: output_dir\n"
          ]
        },
        {
          "output_type": "error",
          "ename": "SystemExit",
          "evalue": "ignored",
          "traceback": [
            "An exception has occurred, use %tb to see the full traceback.\n",
            "\u001b[0;31mSystemExit\u001b[0m\u001b[0;31m:\u001b[0m 2\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.10/dist-packages/IPython/core/interactiveshell.py:3561: UserWarning: To exit: use 'exit', 'quit', or Ctrl-D.\n",
            "  warn(\"To exit: use 'exit', 'quit', or Ctrl-D.\", stacklevel=1)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **Set up Line Drawing Library**\n",
        "\n",
        "The biggest development for me here was parsing through the test.py file to edit the data directory being used and the number of files we would iterate through.\n",
        "- The edited area of the test.py file is labeled\n",
        "- The files are then copied over to the google drive."
      ],
      "metadata": {
        "id": "x6h2ZLl8UNQQ"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Git clone command\n",
        "git_clone_command = \"git clone https://github.com/carolineec/informative-drawings.git\"\n",
        "\n",
        "# Run the command\n",
        "subprocess.run(git_clone_command, shell=True, check=True)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "VoryIKCCTQHP",
        "outputId": "e85b43fd-9443-40e5-f842-0427edf0baad"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "CompletedProcess(args='git clone https://github.com/carolineec/informative-drawings.git', returncode=0)"
            ]
          },
          "metadata": {},
          "execution_count": 4
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# URL to model download.\n",
        "checkpoint_url = 'https://s3.us-east-1.wasabisys.com/visionlab-projects/transfer/informative-drawings/model.zip'"
      ],
      "metadata": {
        "id": "0jWwYtU5UqrX"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def download_and_unzip_model(checkpoint_url, destination_folder):\n",
        "    # Create the destination folder if it doesn't exist\n",
        "    os.makedirs(destination_folder, exist_ok=True)\n",
        "\n",
        "    try:\n",
        "        # Download the model checkpoint zip file\n",
        "        subprocess.run(['wget', '-c', checkpoint_url, '-O', os.path.join(destination_folder, 'model.zip')], check=True)\n",
        "\n",
        "        # Unzip the model checkpoint\n",
        "        subprocess.run(['unzip', os.path.join(destination_folder, 'model.zip'), '-d', destination_folder], check=True)\n",
        "\n",
        "        # Move the contents of the 'model' folder to the destination\n",
        "        model_folder = os.path.join(destination_folder, 'model')\n",
        "        for item in os.listdir(model_folder):\n",
        "            s = os.path.join(model_folder, item)\n",
        "            d = os.path.join(destination_folder, item)\n",
        "            if os.path.isdir(s):\n",
        "                shutil.move(s, d)\n",
        "\n",
        "        # Remove the now-empty 'model' folder\n",
        "        os.rmdir(model_folder)\n",
        "\n",
        "    except subprocess.CalledProcessError as e:\n",
        "        # Handle any errors (non-zero exit code)\n",
        "        print(f\"Error executing command: {e}\")\n",
        "        print(f\"Command output (stderr): {e.stderr}\")\n",
        "\n",
        "# Example: Download and unzip the model\n",
        "checkpoint_url = 'https://s3.us-east-1.wasabisys.com/visionlab-projects/transfer/informative-drawings/model.zip'\n",
        "destination_folder = '/content/informative-drawings/checkpoints'\n",
        "download_and_unzip_model(checkpoint_url, destination_folder)"
      ],
      "metadata": {
        "id": "K3VnZ-6M3Nch"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **Setting up Arguments for Test.py**\n",
        "\n",
        "This involves iterating through each folder in the dataset, and creating a matching folder of the converted line drawings in the results folder."
      ],
      "metadata": {
        "id": "uJTSk8Qgmccg"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Check if a results directory exists. If not, create one.\n",
        "results_dir_name = \"AugmentedDataset\"\n",
        "results_dir_path = os.path.join(OUTPUT_DIR, results_dir_name)\n",
        "os.makedirs(results_dir_path, exist_ok=True)\n",
        "\n",
        "results_root = os.path.join(OUTPUT_DIR, results_dir_name)"
      ],
      "metadata": {
        "id": "vMFxN6RQmyVk"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def convert_images(data_dir, results_root, trainval):\n",
        "\n",
        "    # Create data directory to pull from (either train or val datset)\n",
        "    data = os.path.join(data_dir, trainval)\n",
        "\n",
        "    # Create list of folders in dataset to iterate through.\n",
        "    folders = os.listdir(data)\n",
        "    print('Folders to iterate through:', folders)\n",
        "\n",
        "    # Iterate through folders, running test.py on each.\n",
        "    for folder in folders:\n",
        "        print('\\nConverting images in folder:', folder)\n",
        "        dataroot = os.path.join(data, folder)\n",
        "        files = glob(os.path.join(dataroot, \"*.JPEG\"))\n",
        "\n",
        "        results_root = os.path.join(data_dir, 'AugmentedDataset_2', trainval)\n",
        "\n",
        "        # Create results directory with same name\n",
        "        folder_path = os.path.join(results_root, folder)\n",
        "        os.makedirs(folder_path)\n",
        "\n",
        "        # Set results_dir to the folder that matches the subfolder from the dataset\n",
        "        results_dir = os.path.join(results_root, folder)\n",
        "\n",
        "        # Set number of files to convert to line drawings\n",
        "        how_many = len(files)-1\n",
        "\n",
        "        # Run test.py with these parameters\n",
        "        script_path = 'informative-drawings/test.py'\n",
        "        subprocess.run([\"python\", script_path, \"--name\", \"anime_style\", \"--dataroot\", dataroot, \"--results_dir\", results_dir, \"--how_many\", str(how_many)])\n",
        "\n",
        "    return 0"
      ],
      "metadata": {
        "id": "fQlKyQWBpaFe"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Convert Val dataset\n",
        "convert_images(IMAGENET_ROOT_DIR, results_root, 'val')"
      ],
      "metadata": {
        "id": "GowX47B74JrH"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Convert train dataset\n",
        "convert_images(IMAGENET_ROOT_DIR, results_root, 'train')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 327
        },
        "id": "A5wwvz557dbi",
        "outputId": "6c08cd49-70e8-4260-8862-405afd78bd66"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "error",
          "ename": "FileNotFoundError",
          "evalue": "ignored",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mFileNotFoundError\u001b[0m                         Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-9-186140682505>\u001b[0m in \u001b[0;36m<cell line: 2>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0;31m# Convert train dataset\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 2\u001b[0;31m \u001b[0mconvert_images\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mIMAGENET_ROOT_DIR\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mresults_root\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m'train'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
            "\u001b[0;32m<ipython-input-8-70c5306db9c3>\u001b[0m in \u001b[0;36mconvert_images\u001b[0;34m(data_dir, results_root, trainval)\u001b[0m\n\u001b[1;32m      5\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      6\u001b[0m     \u001b[0;31m# Create list of folders in dataset to iterate through.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 7\u001b[0;31m     \u001b[0mfolders\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mos\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mlistdir\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdata\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      8\u001b[0m     \u001b[0mprint\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'Folders to iterate through:'\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mfolders\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      9\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mFileNotFoundError\u001b[0m: [Errno 2] No such file or directory: '/content/drive/MyDrive/Classes/Third/PSY2356R/ThomasGarity/LineDrawing/datasets/imagenette2-320/train'"
          ]
        }
      ]
    }
  ]
}