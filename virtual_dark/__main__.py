import argparse
import glob

import negative

parser = argparse.ArgumentParser(description="Process negatives")

parser.add_argument("input_imgs", metavar="img", type=str, nargs="+", help="Input images")

args = parser.parse_args()
input_images = args.input_imgs

input_images = map(glob.glob, input_images)
input_images = [item for sublist in input_images for item in sublist] # flatten


print("Input images:\n\t" + "\n\t".join(map(str, input_images)))

for image in input_images:
    neg = negative.from_path(image)
    neg = negative.fully_process_neg(neg)
    neg.show()
