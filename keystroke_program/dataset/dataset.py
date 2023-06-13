import pandas as pd


class Dataset:
    def __init__(self, data):
        self.value = data

        # dataset defined in dataset_modeling()
        self.dataframe = pd.DataFrame

    def dataset_modeling(self):
        # Preparation of the associated dataframe

        if self.value == 1:
            # DNS2009
            dataset = pd.read_excel("dataset\\dns2009\\Dns2009_StrongPasswordData.xls")
            self.dataframe = pd.DataFrame(dataset)

        elif self.value == 2:
            # Greyc-nislab
            dataset = pd.read_excel("dataset\\greyc-nislab\\GREYC-NISLABKeystrokeBenchmarkDatasetSyed.xlsx",
                                    sheet_name='P1')
            self.dataframe = pd.DataFrame(dataset)

            column = "caracteristica"

            # keystroke vector division
            for i in range(len(self.dataframe)):
                mess_until_space = ""
                counter = 1
                vector = self.dataframe["Keystroke Template Vector"][i]
                for j in range(0, len(vector)):
                    if vector[j] == " " and len(mess_until_space) >= 1:
                        self.dataframe.loc[i, column + str(counter)] = float(mess_until_space)
                        counter += 1
                        mess_until_space = ""
                    else:
                        mess_until_space += vector[j]

                self.dataframe.loc[i, column + "64"] = float(mess_until_space)

        elif self.value == 3:
            # Mobikey
            dataset = pd.read_excel("dataset\\mobikey\\Mobikey_dataframe1.xlsx")
            self.dataframe = pd.DataFrame(dataset)

        else:
            # Mobikey temporal
            dataset = pd.read_excel("dataset\\mobikey\\Mobikey_dataframe1_temporal.xlsx")
            self.dataframe = pd.DataFrame(dataset)

    # dataframe transformations, new typing attributes or new users typing values for existing attributes

    def insert_row(self, row):
        if type(row) != list:
            raise ValueError("Sorry, parameter type is not list")

        new_dataframe = pd.DataFrame(row)
        if self.dataframe.empty:
            self.dataframe = new_dataframe
        else:
            if len(row) != len(self.dataframe.columns):
                raise ValueError("Sorry,the length of your row does not match with the length of the columns")

            self.dataframe = self.dataframe.append(new_dataframe, ignore_index=True)

    def insert_column(self, name, column):

        if type(column) != list:
            raise ValueError("Sorry, parameter type is not list")

        if self.dataframe.empty:
            self.dataframe[name] = column

        else:
            if name in self.dataframe.columns:
                raise ValueError("Sorry, column name already contained")

            if len(column) != len(self.dataframe):
                raise ValueError("Sorry, the length of your column does not match with the length stored")

            self.dataframe[name] = column

    # getters and setters

    @property
    def value_getter(self):
        return self.value

    @value_getter.setter
    def value_getter(self, new_value):
        if new_value != 1 and new_value != 2 and new_value != 3:
            raise ValueError("Sorry, your value does not accomplish the criteria")
        self.value = new_value

    @property
    def dataframe_getter(self):
        return self.dataframe

    @dataframe_getter.setter
    def dataframe_getter(self, new_dataframe):
        if not isinstance(new_dataframe, pd.DataFrame): 
            raise ValueError("Sorry, your value does not accomplish the criteria")

        self.dataframe = new_dataframe
