# Object Tracking On Fish-eye Image

#### Installation 

#### Basic usage



###### a. Undistort images

​	Here, we support two datasets, one is the **oxford RobotCar dataset** and the other is **ROS bags.** 

​	[RobotCar Dataset](https://github.com/ori-mrg/robotcar-dataset-sdk) for Oxford RobotCar

​	tools.py for ros bags, for example:

```sh
python tools.py --bag 2019-11-13-17-16-19-highway-fov150-yaw.bag --calib_dir ~/calib/FOV150_red --output j7-8L4E-sensor 
```

​		where `--calib_dir` is the directory where puts chessboard images.

###### b. do obstacle detection and save to csv file

​	Open source detection model is used here.  

​	refer to https://github.com/xingyizhou/CenterNet

​	And the detection result should be saved as following csv format:

​	`img,id,class,tlx,tly,brx,bry,score`

​	some dump code.

```python
def main(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = max(opt.debug, 0)
    Detector = detector_factory[opt.task]
    detector = Detector(opt)

    if os.path.isdir(opt.demo):
        image_names = []
        ls = os.listdir(opt.demo)
        for file_name in sorted(ls):
            ext = file_name[file_name.rfind('.') + 1:].lower()
            if ext in image_ext:
                image_names.append(os.path.join(opt.demo, file_name))
    else:
        image_names = [opt.demo]
    with open(output_file, "w") as f:
        f.write("img,id,class,tlx,tly,brx,bry,score\n")
        for (image_name) in image_names:
            print(image_name)
            ret = detector.run(image_name)
            detection = ret['results']
            idx = 0
            for key, items in detection.items():
                for i in range(items.shape[0]):
                    line = image_name + "," + str(idx) + "," + str(key) + "," + str(items[i, 0]) + "," + str(items[i, 1]) + "," + str(items[i, 2]) + "," + str(items[i, 3]) + "," + str(items[i, 4]) + "\n"
                    f.write(line)
                    idx += 1

```



###### c. estimate 3-d position

```sh
python mono_dist.py --dir oxford/2014-06-24-14-15-17 --output oxford/2014-06-24-14-15-17/mono_left_dist
```

```sh
python mono_dist.py --dir /media/andy/jinwen-2TB/j7-8L4E-sensor/2019-11-13-17-16-19-highway-fov150-yaw --calib_path /media/andy/jinwen-2TB/j7-8L4E-sensor/calib --is_plusai true --fov 150 
```

**calib_path** stores calibration files. 

###### e. tracking object

**TODO**