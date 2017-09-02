# Vehicle-Detection-Project5
In this project, computer vision techniques are used to accurately detect vehicles in a video and track them. 
The video processing pipeline ``process_vid()``, consists of the following steps:
  1. Receiving input frame as an RGB image
  2. Extract features from the image, scaled in 4 different sizes: in 64x64 windows that are predicted to be cars, using `detect_vehicle()` function.
  3. Generate a heatmap using the predicted windows using the `generate_heatmap()` function, using only pixels that are higher than the specified `threshhold`.
  4. Use the `label()` function from the library SciPy's `scipy.ndimage.measurements` module, to group each group of overlapping windows into single windows.
  5. Assume each labeled window is a possible vehicle, and calculate it's dimensions (centers, width, height), and add to `local_detections` as an object of the class `Vehicle()`.
  6. Loop over the previously detected windows, and the currently detected and if both have close centers, assume that this detection resembles a new car: append this `Vehicle()` to the global array `cars`.
  7. Loop over the detected `cars`, for every `car`, loop over the `local_detections`:
      1. If the centers and dimensions of the car and detection are close enough, do not save this detection for the next frame and assume its a car; `car.detected = True`.
      2. Count how many times has this car been detected `car.n_detected += 1` and `car.not_detected -= 1`.
      3. If the car wasn't similar to any of the detected windows, increment `car.not_detected += 1`.
  8. Using the number of frames the car has been detected or not, we can approach a decision:
    * Either: Draw the window detected since it most probably resembles a car.
    * Or: Remove the detected window since it was presumably a false positive.
  9. The average of widths and heights are used to smooth the window drawn to make sure there aren't any abrupt changes in window's dimensions.
      
##### Check this [video](./project_output.mp4) to preview the pipeline's output
##### Here is a [youtube link](https://youtu.be/3YX-kcZqPTE) for my video to stream it online.

*Please check the [writeup report](./writeup_report.md) for further details*
*Also check my implementation contained in this [IPython Notebook](./Advanced_LaneFinding_Project-Process-Notebook.ipynb)
