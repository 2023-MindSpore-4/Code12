import mindspore as ms
from mindspore import ops


def make_image_grids(img_tensor, n_cols):
    batch_size, n_caps, c, h, w = img_tensor.shape
    row_list = []
    index = 0
    while index < n_caps:
        col_list = []
        for j in range(n_cols):
            if index >= n_caps:
                img = ops.zeros([batch_size, c, h, w])
            else:
                img = img_tensor[:, index]
            if j == n_cols-1:
                col_list.append(img)
            else:
                col_list.append(ops.cat([img, ops.ones([batch_size, c, h, 1])], axis=-1))
            index += 1
        row_list.append(ops.cat(col_list, axis=-1))
    img_array = ops.cat(row_list, axis=-2)
    return img_array


def transform_img_tensor(img_tensor):
    img_tensor = img_tensor.value()
    # img_tensor = (img_tensor + 1.) / 2.
    # [batch_size, n_caps, C, H, W]
    if len(img_tensor.shape) == 5:
        if img_tensor.shape[2] == 4:
            img_tensor = img_tensor[:, :, :3] * img_tensor[:, :, 3:4]
        img = make_image_grids(img_tensor, 6)
        return img
    # [batch_size, C, H, W]
    elif len(img_tensor.shape) == 4:
        if img_tensor.shape[1] == 4:
            img_tensor = img_tensor[:, :3] * img_tensor[:, 3:4]
        img_array = img_tensor.numpy()
        return img_array
