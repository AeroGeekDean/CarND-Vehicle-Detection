## Writeup

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.jpg
[image2]: ./output_images/HOG_example.jpg
[image3]: ./output_images/sliding_windows_1.jpg
[image4]: ./output_images/sliding_windows_2.jpg
[image5]: ./output_images/Moving_Pictures.jpg

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Code layout

Overall, the code base is a collection of functions within the file [`'lesson_functions.py'`](./lesson_functions.py). These functions were taken originally from the course lesson examples, and modified to suit the needs of this project.

The notional 'main()' portion of the project is the Jupyter notebook file [`'P5-Project_notebook.jpynb'`](./P5-Project_notebook.jpynb), which imports the functions from `'lesson_functions.py'` and describes the steps taken for:
- feature extractions from train/test images
  - spatial down-sample binning
  - colorspace histogram
  - HOG
- input normalization
- train/test data split
- model (Linear SVM Classifier)
  - training
  - score evaluation

For processing on a road image, the code further do the following (see **Pipeline section** later on):
- sliding window search (for cars) across road image
- create heatmap based on search results
- processing of time-series of images (ie: video)
  - apply threshold on current image heatmap ('static threshold')
  - setup **circular buffer** to track previously frame heatmaps
  - apply **decaying** feature to the circular buffer
  - apply threshold on historic total heatmap within the buffer ('dynamic threshold')
  - search for contiguous heat regions
  - draw final boxed images

### Histogram of Oriented Gradients (HOG)

#### HOG Feature Extraction

The `'get_hog_features()'` function within `'lesson_functions.py'` contains the HOG code. It is called by 'Feature Extraction' section of the Jupyter notebook.

Both vehicle (`positive`) and non-vehicle (`negative`) images were read in.  Here is an example of one of each of the `Car` and `Not-Car` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

#### HOG (and non-HOG) parameters.

I tried various combinations of parameters with a simple prototyping pipeline setup to train the classifier and then detect vehicles in (full driver's view) test images. The following parameters gave me the best adequate results:
> with training images at (64 x 64) pixels:
- Colorspace = YCrCb
- Spatial down-sample binning = (32 x 32)
- Colorspace histogram bins = 32 per channel
- HOG channels = ALL
- HOG orientation = 9
- HOG pixel_per_cell = (8 x 8)
- HOG cell_per_block = (2 x 2)


### Classifier Training

I first normalized the feature vector (`X`) with `sklearn.preprocessing.StandardScaler`, then trained a Linear SVM Classifier (`sklearn.svm.LinearSVC`) using SKLearn's standard API of `fit()`. I also split out the data for training / testing with a percentage proportion of 80/20.

The trained classifier achieved an accuracy of 99.52%

### Sliding Window Search

The sliding window search was implemented in the `'find_cars()'` function within the [`'lesson_functions.py'`](./lesson_functions.py) file.

This originally came from [instructor Ryan Keenan's Project 5 walk thru Q&A video](https://www.youtube.com/watch?v=P2zwrTM8ueA), and also from the class room lessons (Lesson #35 - Hog Sub-sampling Window Search). But the calculation of the cell/block numbers and steps were not very clear. Thus I went thru and re-wrote / commented the code in a fashion that is more clear for me to follow.

From experimentation, using **window scaling sizes** of `[0.5, 0.75, 1.0, 1.5, 2.0]` the algorithm were able to identify cars on the test image. However, the smaller window sizes resulted in a LOT of searches per image frame, **on the order of over 1000!** Thus I settled for scale sizes of `[1.0, 2.0]` for the project, as a computational performance tradeoff.

I used a **window overlap of 75%**, thus it takes 4 sliding steps for the window to cover a completely new image area.

My code is listed below:

```Python
# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,
              cell_per_block, spatial_size, hist_bins, include_img=True):
    bbox_list = []

    # force input img to be of range (0,255) 'uint8', for consistency sake
    if np.max(img)<=1.0:
        img = img*255
        img = img.astype(np.uint8)

    if include_img:
        draw_img = np.copy(img)

    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')

    # For computational efficiency, HOG of entire image is taken in 1-shot, then cropped down
    # for the window search area. And because training was done on 64x64 pixel image size, if the
    # search window is larger, we need to sub-sample the image such that 64x64 pixels fit back in
    # the window. Thus 'scale' is size of search window RELATIVE TO training image size.
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Compute individual channel HOG features for the entire image
    # NOTE - return shape = [nblocks x nblocks  x  cell_per_block x cell_per_block  x  orient]
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    # number of cells across entire image
    nxcells = (ch1.shape[1] // pix_per_cell)
    nycells = (ch1.shape[0] // pix_per_cell)

    # number of HOG blocks across entire image. HOG blocks walk 1 cell step at a time.
    nxblocks = nxcells - cell_per_block + 1
    nyblocks = nycells - cell_per_block + 1

#     nfeat_per_block = orient*cell_per_block**2 # not used

    # 64x64 was the orginal sampling window size used for training, with 8 cells and 8 pix per cell
    train_window_size = 64 # pixels

    nblocks_per_window = (train_window_size // pix_per_cell) - cell_per_block + 1

    # Instead of overlap, define how many cells in the search window step
    cells_per_step = 2

    # number of search window patches (steps) across entire image
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    for xb in range(nxsteps):
        xpos = xb*cells_per_step
        xleft = xpos*pix_per_cell

        for yb in range(nysteps):
            # xpos = xb*cells_per_step
            ypos = yb*cells_per_step

            # Extract HOG features for this patch (from earlier HOG output of entire image)
            # Recall HOG output shape = [nblock x nblock  x  cell_per_block x cell_per_block  x  orient]
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            # For other features, we need actual pixel positions
            # xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+train_window_size, xleft:xleft+train_window_size],
                                (train_window_size,train_window_size))

            # Get color & hist features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # build full feature vector
            features = np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)

            # Scale features and make a prediction
            test_features = X_scaler.transform(features)
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                # resize back to orig image size
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(train_window_size*scale)

                corner1 = (xbox_left, ytop_draw+ystart)
                corner2 = (xbox_left+win_draw, ytop_draw+win_draw+ystart)

                # could return list of window bounding boxes instead.
                bbox_list.append((corner1, corner2))

                if include_img:
                    cv2.rectangle(draw_img, corner1, corner2, (0,0,255), 6) # blue box

    if include_img:
        nsearches = nxsteps*nysteps # number of searches conducted
        return bbox_list, draw_img, nsearches
    else:
        return bbox_list
```

Below are examples of sliding window search results for scale sizes of 1.0 and 2.0, along with their heatmap.

`11 bounding boxes found, out of 912 searches at 1.0 scale level`
![alt text][image3]

`5 bounding boxes found, out of 144 searches at 2.0 scale level`
![alt text][image4]


Not to be confused with... [click me](https://www.youtube.com/watch?v=HV36oKAwGKQ) for some awesomeness!
##### "Sliding windows" -vs- "Moving Pictures"
![Rush: Moving Pictures][image5]

---

### Video Implementation

Here's a [link to my video result on YouTube](https://www.youtube.com/watch?v=PaB5t5AFIkc)

#### Video Processing Pipeline

##### Static Thresholding
For each frame of the video, bounding boxes are identified for each search-window positive detections. For true-positive detection, there are often many overlapping bounding boxes, from which a heatmap is created. The heatmap is then 'static' thresholded to remove low-confidence areas as potential false-positives.

##### Decaying Circular-Buffer
A circular-buffer was setup using `'collections.deque'` to store the last few X-frames of heatmap. Additionally, the stored past heatmaps are decayed (ie: 'cooled') at each frame.

##### Dynamic Thresholding
Next, the entire buffer of stored heatmaps (with decay), are totaled to come up with a **'total' heatmap** of vehicle positions. This 'total' heatmap is also 'dynamic' thresholded to help remove dynamic false-positives that are artifacts of previous 'pop-up' true-positive detections.

This is equivalent to a weighted-time-averaging scheme where the more recent results have higher weight. (Mentally picture the 'primary return' decaying blips of a radar scope...)

>Example:
A positive detection of vehicle passing-by in the on-coming lane would cause a decaying residual hot spot. The 'dynamic' thresholding will help remove this residual hot spot prior to having to wait for it to fully decay to zero.

>Conversely, only recent hot spots that stay within a vicinity will pass threshold and be confidently labeled as an identified vehicle being tracked.

##### Contiguous Region labeling
Finally, `scipy.ndimage.measurements.label()` was used to identify individual blobs in the final heatmap. Each blob is assumed to be corresponded to a vehicle. And final bounding boxes are constructed to cover the area of each blob detected.

Below is the code that does this:

```Python
import collections

buffer_length = 10
heatmap_history = collections.deque(maxlen=buffer_length)

# Note heatmap_history is external to the function below, thus is effectively a static variable

def find_vehicles(img):
    scales = [1.0, 2.0]
#     scales = [0.75, 1.0, 1.5, 2.0] # orig planned scales to use
    threshold_static = 1
    threshold_dynamic = 0
    decay_rate = 1
    max_val = 255 # for uint8

    # init a new 'cold' heatmap
    heatmap = np.zeros(img.shape[:2]).astype(np.uint8)

    # loop thru all the search window sizes
    for scale in scales:
        bbox_list = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,
                              cell_per_block, spatial, histbin, include_img=False)
        heatmap = add_heat(heatmap, bbox_list)

    heatmap = apply_threshold(heatmap, threshold_static)
    heatmap = np.clip(heatmap, 0, max_val) # max value of int8

    # decay the existing heatmap history, before adding latest heatmap
    for hm in heatmap_history:
        hm[hm>0] -= decay_rate
    heatmap_history.append(heatmap)

    heatmap_total = np.sum(heatmap_history, axis=0)

    # search for contiguous heat regions
    labels = label(heatmap_total) # labels is a tuple

    # draw final boxed image
    draw_img = draw_labeled_bboxes(img, labels)

    return draw_img
```

---

### Discussion

The approach for this project was pretty straight forward. The biggest problem was trying to identify:

1. optimal Computer Vision (HOG) parameters for each frame, to avoid static false positives caused by trees, signs, shadows and other non-vehicle items.

2. optimal parameters for the time-dependent effect of decaying circular buffers and static + dynamic thresholds (discussed above). This was aimed more towards quickly eliminating residual effects of correctly identifying fleeting objects (such as vehicle in the on-coming lanes), without sacrificing performance of true-positives of vehicles nearby.

The time-dependent parameter tweaking was time consuming due to the long processing time for the 50sec-long video. Each processing iteration took about ~30 minutes on my computer (Late-2012 era Mac Mini with Intel Quad Core i7). I made about 15 iterations before choosing result from one of the earlier iterations.
