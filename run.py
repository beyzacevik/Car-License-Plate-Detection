from visualize import *
import sys
if __name__ == "__main__":

        try:
                dataset_path = sys.argv[1]
                original_path = sys.argv[2]
                annotation_path = sys.argv[3]

                Visualize = visualize(dataset_path, original_path, annotation_path)
                Visualize.execute()
        except Exception as e:
                print(e)
                print("Wrong number of arguments")

