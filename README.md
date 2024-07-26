# Video LLAVA SGG


# Change logs


V12:
- Frame wise response [0-7]. 
    - :white_check_mark: list objects per frame {0: obj1,obj2, 1: obj1,obj3}
    - :white_check_mark: list triplets per frame {0: <subj,pred,obj>, 1: <subj,pred,obj>}
    - :black_square_button: seperate instructions for bounding box "provide location for obj1"
- :black_square_button: apply V11 changes to frame/image sg code.

V11:
- :white_check_mark: Added implementation for selected indices to pass to model for training and testing.
- :white_check_mark: Select frames which covers more relations, this will help reducing the frames to 8 

V10:
- :white_check_mark: updated frame selection logic same as video-llava code.

V9 Changes:
- :white_check_mark: Only uni-index annotations should be taken for video
- :white_check_mark: functions improvements for better reuse of code. (e.g conversation template)

V8 Changes:
- :white_check_mark: for list of objects, added frame idx[0, 7] since video-llava takes 8 frames for the input video.
    - This will help when the object is visible in the video from start to end as acts as a temporal entity
    - object may be visible on all frames, but taken only first occurance when asked for a list of objects to avoid redundancy.
- :black_square_button: for relation of objects, add nearest frame_idx from [0,7] and its respective bounding box
    - here also multiple frames are possible from [0,7] but only the first occurance is taken. 
- :white_check_mark: make vid sg annotation threaded

V7 Changes:

- :white_check_mark: child,sitting on,chair ==> child:sitting on:chair for easy parsing the results
- :black_square_button: Instead of average frame add frame start and end normlized
    - Above will not solve the temporal prediction problem
    - :white_check_mark: Added frame wise annotations since Video-LLAVA takes only 8 frames from the video. This will be image-fine-tune dataset
      - issue: large number of frames, takes a lot of time to train

- :white_check_mark: - instead of taking all the frames from start to end taking only min max of annotations


V6:
- :white_check_mark: Normlized Temporal frame value
- :white_check_mark: bb values are rounded upto 3 decimal
    - issue : predicting wrong frame (second frame can be added)

V5:
- :white_check_mark: removed frame size in response.
- :white_check_mark: removed frame from the objects list
- :white_check_mark: removed Q&A pairs from pvsg dataset
- :white_check_mark: removed summary from the pvsg dataset

V4:
- :white_check_mark: Normlize BB
- :white_check_mark: Seperate List of objects and Relationship between the objects (experimental) 
    - :white_check_mark: Reverting back to check the issues, results are not good (v5)
- :white_check_mark: Keep same questions for all conversation
- :white_check_mark: If BB is not found for any objects keep the list empty instead of passing zeros.
- :white_check_mark: Removed list of categories before providing summary
- :white_check_mark: Fix for multiple frames objects might appeare e.g adult.1_[0,10], adult.1_[30,60]
- :white_check_mark: Taken avg frame for gettig bb for an object e.g adult.1_[0,10] => 5th frame will be taken to get BB
    - issue: sometimes in the average frame the object is occluded which results in no BB for the object.