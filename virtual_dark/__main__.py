import argparse
import glob

from virtual_dark import negative

parser = argparse.ArgumentParser(description="Process negatives")

parser.add_argument("--output_dir", metavar="out_dir", type=str, nargs="?", help="Output directory", default="output")
parser.add_argument("--show_after", help="Whether to show image after processing", action="store_true")
parser.add_argument("input_imgs", metavar="img", type=str, nargs="+", help="Input images")

args = parser.parse_args()
input_images = args.input_imgs
output_dir = args.output_dir
show_after = args.show_after

input_images = map(glob.glob, input_images)
input_images = [item for sublist in input_images for item in sublist]  # flatten

print("Input images:\n\t" + "\n\t".join(map(str, input_images)))
print("Outputting to {}".format(output_dir))

total_images = len(input_images)
curr_image = 1

for image in input_images:
    print("Processing: {}".format(image), "{}/{} images".format(curr_image, total_images))
    neg = negative.from_path(image)
    neg = negative.fully_process_neg(neg)
    curr_image += 1
    if show_after:
        neg.show()
    neg.save_to_dir(output_dir)
