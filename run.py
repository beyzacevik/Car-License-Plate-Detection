from visualize import *
import sys
if __name__ == "__main__":

        try:
                dataset_path = sys.argv[1] #'/Users/beyzacevik/Downloads/Dataset'
                original_path = sys.argv[2] #'/Users/beyzacevik/Downloads/Dataset/Original_Subset'
                detection_path = sys.argv[3] #'/Users/beyzacevik/Downloads/Dataset/Detection_Subset'

                Visualize = visualize(dataset_path, original_path, detection_path)
                Visualize.execute()
        except Exception as e:
                print(e)
                print("Wrong number of arguments")

