# %%
import torch


class RankAxesShape():
    def __init__(self):
        self.dd = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]

    def rank_of_tensor(self):
        """
        A tensor's rank tells us how many indexes are needed to refer to a specific element within the tensor,
        the number of dimensions present within the tensor
        """
        print(self.dd[0][0])
        print("Rank 2")

    def axes_of_tensor(self):
        """
        An axis of a tensor is a specific dimension of a tensor.The length of each axis tells us
        how many indexes are available along each axis.
        """
        print("Each element along the first axis, is an array: ")
        print(self.dd[0])
        print(self.dd[1])
        print(self.dd[2])
        print("Each element along the second axis, is a number: ")
        print(self.dd[0][0])
        print(self.dd[1][0])
        print(self.dd[2][1])

    def shape_of_tensor(self):
        """The shape of a tensor gives us the length of each axis of the tensor. """
        print(torch.tensor(self.dd).shape)


s = RankAxesShape()
s.rank_of_tensor()
s.axes_of_tensor()
s.shape_of_tensor()

# %%
