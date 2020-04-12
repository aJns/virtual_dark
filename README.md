# A script/program to automate film negative processing

# Features
- [ ] Find, crop and rotate negatives
    - [x] find & rotate
- [x] Invert colors
- [ ] Correct color profile

# Workplan
1. Find the individual negatives
    * Use the film strip holes for this
2. Crop and rotate them
    * Should be easy after 1.
    * Cropping is not so easy to automate, I think; Maybe manual cropping for
      now, until I get around to it
3. Correct with whitepoint
    * Find a piece of blank film, set that as the white point
    * This was a good idea, but not enough; Proper color correction requires
        1. Color curve adjustment, so that for each color channel highs and lows
           in the actual image match. Ie for red for example, in the negative
           there are values from 0 to 255
        2. Color balance adjustment, white point stuff I think
4. Inverting colors should also be simple

# Color correcting with curves
1. Find image area
2. In the area, find the highest and lowest value
3. Map all values so that the highest and lowest value are 255 and 0
   respectively

I think this can be done before or after inversion
