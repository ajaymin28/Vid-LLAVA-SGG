# Video LLAVA SGG


# Change logs
```
V11
[X] Added implementation for selected indices to pass to model for training and testing.
[X] Select frames which covers more relations, this will help reducing the frames to 8

V10
[X] updated frame selection logic same as video-llava code.

V9 Changes
[X] Only uni-index annotations should be taken for video
[X] functions improvements for better reuse of code. (e.g conversation template)

V8 Changes
[X] for list of objects, added frame idx[0, 7] since video-llava takes 8 frames for the input video.
    - This will help when the object is visible in the video from start to end as acts as a temporal entity
    - object may be visible on all frames, but taken only first occurance when asked for a list of objects to avoid redundancy.
[] for relation of objects, add nearest frame_idx from [0,7] and its respective bounding box
    - here also multiple frames are possible from [0,7] but only the first occurance is taken. 
[X] make vid sg annotation threaded

V7 Changes
[X] child,sitting on,chair ==> child:sitting on:chair for easy parsing the results
[] Instead of average frame add frame start and end normlized
    - Above will not solve the temporal prediction problem
    - [X] Added frame wise annotations since Video-LLAVA takes only 8 frames from the video. This will be image-fine-tune dataset
      - issue: large number of frames, takes a lot of time to train

[X] - instead of taking all the frames from start to end taking only min max of annotations


V6 Changes

[x] Normlized Temporal frame value
[x] bb values are rounded upto 3 decimal
    - issue : predicting wrong frame (second frame can be added)

V5 Changes

[x] removed frame size in response.
[x] removed frame from the objects list
[x] removed Q&A pairs from pvsg dataset
[x] removed summary from the pvsg dataset

V4 Changes

[X] Normlize BB
[X] Seperate List of objects and Relationship between the objects (experimental) 
    - [X] Reverting back to check the issues, results are not good (v5)
[X] Keep same questions for all conversation
[X] If BB is not found for any objects keep the list empty instead of passing zeros.
[X] Removed list of categories before providing summary
[X] Fix for multiple frames objects might appeare e.g adult.1_[0,10], adult.1_[30,60]
[X] Taken avg frame for gettig bb for an object e.g adult.1_[0,10] => 5th frame will be taken to get BB
    -issue: sometimes in the average frame the object is occluded which results in no BB for the object.
```