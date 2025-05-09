To download and compile, run `bash setup_ffmepg.sh`.

It will create a directory called ffmpeg-3.4.8.
After that, will be necessary configure env variables.

`vim .bashrc`
Add the following two lines in the end of the file
```
export LD_LIBRARY_PATH="/home/user/ffmpeg-3.4.8/build/lib/"
export PATH=$PATH:"/home/user/ffmpeg-3.4.8/build/bin/"
```
Just change the user with your user name.