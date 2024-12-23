import os
from PIL import Image


def save_result(outputs_list, save_dir, img_names):
    output = outputs_list[-1]

    for i in range(output.shape[0]):
        img_name = img_names[i]
        output_i = output[i].squeeze().cpu().detach().numpy()
        output_i = output_i * 255
        output_i = output_i.astype('uint8')
        output_img = Image.fromarray(output_i.squeeze(), mode='L')

        # Save the image
        filename = f'result_{img_name}.png'
        output_path = os.path.join(save_dir, filename)
        output_img.save(output_path)
