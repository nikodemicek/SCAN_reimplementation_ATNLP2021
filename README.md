# Generalization without systematicity

This is a reimplemenation project of the Lake & Baroni 2018 [paper](https://arxiv.org/abs/1711.00350).

By default running the notebook without any change of parameters will test the best model for exp 3 with primitive jump command split with the transformers with similar parameters to RNNs, but 800 dim_feedforward and 4 attention heads.

## Loading data

First, make sure that the notebook is in the same folder as the data root folder. otherwise it could be necessary to change the paths in Loading Data section.

To load the desired SCAN split, just change the variable in the load_data function call. the paths are pre-written. If you want to test the exp 1 size variations or exp 3 composed commands, you will need to change only the values of variables "percent_of_commands" and "num,rep", respectively.

```python
# for experiment no 1 - simple split and size variations (sv)
simple_train_path = "/content/drive/MyDrive/atnlp/SCAN/simple_split/tasks_train_simple.txt"
simple_test_path = "/content/drive/MyDrive/atnlp/SCAN/simple_split/tasks_test_simple.txt"

percent_of_commands = 1
simple_train_path_sv = f"/content/drive/MyDrive/atnlp/SCAN/simple_split/size_variations/tasks_train_simple_p{str(percent_of_commands)}.txt"
simple_test_path_sv = f"/content/drive/MyDrive/atnlp/SCAN/simple_split/size_variations/tasks_test_simple_p{str(percent_of_commands)}.txt"


#for experiment no 2

length_train_path = "/content/drive/MyDrive/atnlp/SCAN/length_split/tasks_train_length.txt"
length_test_path = "/content/drive/MyDrive/atnlp/SCAN/length_split/tasks_test_length.txt"

#for experiment no 3
lturn_train_path = "/content/drive/MyDrive/atnlp/SCAN/add_prim_split/tasks_train_addprim_turn_left.txt"
lturn_test_path = "/content/drive/MyDrive/atnlp/SCAN/add_prim_split/tasks_test_addprim_turn_left.txt"
jump_train_path = "/content/drive/MyDrive/atnlp/SCAN/add_prim_split/tasks_train_addprim_jump.txt"
jump_test_path = "/content/drive/MyDrive/atnlp/SCAN/add_prim_split/tasks_test_addprim_jump.txt"

#for exp no 3 jump composed
def get_path(num, rep, traintest= 'train'):
    path = f"/content/drive/MyDrive/atnlp/SCAN/add_prim_split/with_additional_examples/tasks_{traintest}_addprim_complex_jump_{num}_{rep}.txt"
    return path

num,rep = 'num32','rep4'
train_composed_jump = get_path(num=num,rep=rep)
test_composed_jump = get_path(num=num,rep=rep,traintest = 'test')

#load the data from the path into variable
train_data = load_data(jump_train_path)
test_data = load_data(jump_test_path)

```


## Running the code

Please run the cells in the notebook in order. You can leave out the 3. Group part. Since the model takes ~ 1 hour to train.

Section 4 transformer runs fast (~3min).

Some of the classes necessary for transformer in the appendix are in section 4 before 4.1 Encoder. The transformer in the appendix runs fast as well.

*note that the code runs fast because it achieves significantly better results in fewer iterations, not because the iterations are faster


