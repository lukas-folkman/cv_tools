# Computer Vision Tools (cv_tools)

## Dependencies
```
conda env create --file envs/cv_tools_minimal.yaml
conda activate cv_tools
```

## Use cases
```
# Activate conda environment
cd cvat
conda activate cv_tools

# Display the help message:
python video_to_CVAT.py --help

# First check video metadata:
python video_to_CVAT.py "/mnt/c/Users/YOUR_USER_NAME/Downloads/video.mp4" --just_info

# Extract frames and create a CVAT task:
python video_to_CVAT.py "/mnt/c/Users/YOUR_USER_NAME/Downloads/video.mp4" --every_n_frame 3 --stop_at 9000 --labels open closed DJ --output_dir "/mnt/c/Users/YOUR_USER_NAME/Downloads" --disable_fast_forward

# Extract frames with automatic annotation and create a CVAT task:
python video_to_CVAT.py "/mnt/c/Users/YOUR_USER_NAME/Downloads/video.mp4" --every_n_frame 3 --stop_at 9000 --labels open closed DJ --output_dir "/mnt/c/Users/YOUR_USER_NAME/Downloads" --annotate "/mnt/c/Users/YOUR_USER_NAME/PhD_Project/Models/WHICH_MODEL/model.pt" --disable_fast_forward
```

```
# Remove images without annotations (permanently)
# NOTE: annotations.json format must be COCO 1.0
python remove_img_without_annot.py --annot_fn annotations.json --img_dir ../my_frames
```

```
# Assuming that $HOME/VideoLAN/VLC is the executable binary of VLC Player
bash transcode_video.sh --vlc $HOME/VideoLAN/VLC video1.asf video2.asf

# Use wild-cards for transcoding a large number of files
bash transcode_video.sh --vlc $HOME/VideoLAN/VLC ../my_videos/*.asf
```

## Shrink your WSL2 virtual disk on Windows

```
wsl --shutdown
diskpart
# open window Diskpart
select vdisk file="C:\Users\YOUR_USER_NAME\AppData\Local\Docker\wsl\data\ext4.vhdx"
attach vdisk readonly
compact vdisk
detach vdisk
exit
```

## `CVAT` on Windows (without ML models)

1. If you like, create a new user
2. Run PowerShell as admin
```
wsl --install --distribution Ubuntu-20.04
wsl --set-default-version 2
wsl --set-default Ubuntu-20.04
```
3. Enable Server service (as admin)
  - Start --> Services --> Server --> Properties --> Startup type --> Automatic
  - Restart the computer and check if Server is running
4. Install docker (as admin)
  - https://www.docker.com/products/docker-desktop/
5. Run Ubuntu
6. Download `CVAT` in Ubuntu
```
git clone https://github.com/opencv/cvat
cd cvat
# git checkout 58b05536f5c1546dfa34d80a911b17e029d33efb
git checkout v2.4.2
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```
7. Create a user
```
sudo docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
```
8. Open CVAT using Chrome: http://localhost:8080/
9. Create docker volume
```
docker volume create --name cvat_share --opt type=none --opt device="$JELLIES_ANNOT_DIR" --opt o=bind
```
10. Shutdown CVAT
```
docker compose down
```
11. Restart with all components
```
bash run-cvat-win.sh
```
12. Open CVAT using Chrome: http://localhost:8080/

13. Optional: Setup SSH keys for GitHub in Ubuntu
```
ssh-keygen -t rsa -b 4096 -C "your_email"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_rsa
cat ~/.ssh/id_rsa
# copy it to Github → Settings → SSH and GPG keys → New SSH key
ssh -T git@github.com  # to test if authentication works
```
15. Optional: Download `cv_tools` in Ubuntu
```
cd "somewhere"
git clone git@github.com:lukas-folkman/cv_tools.git
cd cv_tools
```
16. Update to CVAT v2.4.2 with Segment Anything Model
```
git checkout master
git checkout v2.4.2
# delete all containers in docker
bash run-cvat.sh
bash deploy-sam.sh
# serverless/deploy_cpu.sh serverless/pytorch/facebookresearch/sam
```
