import argparse

parser = argparse.ArgumentParser(description="Process negatives")

parser.add_argument("input_imgs", metavar="img", type=str, nargs="+", help="Input images")

args = parser.parse_args()
input_images = args.input_imgs

print("Input images:\n\t" + "\n\t".join(map(str, input_images)))
