from matplotlib import pyplot as plt
import numpy as np
from keras import backend as K
from settings import N_IMAGES_TO_SHOW, CLASSES_NO


def remove_axes(fig):
    for ax in fig.axes:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


def plot_images(images, size, columns, rows):
    fig = plt.figure(figsize=size)
    for x, i in zip(range(1, columns * rows + 1), images):
        fig.add_subplot(rows, columns, x)
        plt.imshow(i)
    remove_axes(fig)
    plt.show()


# dimensions of the generated pictures for each filter.
img_width = 32
img_height = 32

# util function to convert a tensor into a valid image


def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())

def get_filters(model):
    """
    This code comes almost entirely from keral tutorial page.
    :param model: pretrained model, that filters we'd like to visualise.
    :return:
    """
    input_img = model.input
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    kept_filters = []
    for layer_name in layer_dict.keys():
        if 'conv' not in layer_name:
            continue
        layer_filters = []
        for filter_index in range(15):
            layer_output = layer_dict[layer_name].output
            if K.image_data_format() == 'channels_first':
                loss = K.mean(layer_output[:, filter_index, :, :])
            else:
                loss = K.mean(layer_output[:, :, :, filter_index])
            grads = K.gradients(loss, input_img)[0]
            grads = normalize(grads)
            iterate = K.function([input_img], [loss, grads])
            step = 1.
            if K.image_data_format() == 'channels_first':
                input_img_data = np.random.random((1, 3, img_width, img_height))
            else:
                input_img_data = np.random.random((1, img_width, img_height, 3))
            input_img_data = (input_img_data - 0.5) * 20 + 128

            for i in range(20):
                loss_value, grads_value = iterate([input_img_data])
                input_img_data += grads_value * step
                if loss_value <= 0.:
                    break
            if loss_value > 0:
                img = deprocess_image(input_img_data[0])
                kept_filters.append((img, loss_value))
    layer_filters.sort(key=lambda x: x[1], reverse=True)
    kept_filters += layer_filters[:8]
    images = [k[0] for k in kept_filters]
    images = np.stack(images)
    images = images.astype(np.float32)
    images /= 255
    return images
