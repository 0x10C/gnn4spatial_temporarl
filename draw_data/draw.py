import  os
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    file_list = os.listdir("./")
    file_list = [ele for ele in file_list if "csv" in ele]
    measure_name = ["acc","train_loss","test_loss","f1"]
    for measure in measure_name:
        for file in file_list:
            temp = pd.read_csv(file)
            temp_y = list(temp[measure])
            x = [ele for ele in range(100,100*(len(temp_y)+1),100)]
            plt.plot(x,temp_y,label = file.split(".")[0])
        plt.xlabel('batch')
        plt.ylabel(measure)
        plt.legend()
        plt.savefig("{}.jpg".format(measure))