import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure

def next_length_pow2(x):
    return 2 ** np.ceil(np.log2(abs(x)))

def filter(fft_data, filter_large_dia, filter_small_dia):
    fft_data_shape = fft_data.shape
    side_len = fft_data_shape[0]

    filter_large = 2.0 * filter_large_dia / side_len
    filter_small = 2.0 * filter_small_dia / side_len

    filter_data = np.ones_like(fft_data, dtype=np.float32)

    #calculate factor in exponent of Gaussian from filterLarge / filterSmall
    scale_large = filter_large ** 2
    scale_small = filter_small ** 2

    fft_data = fft_data.flatten()
    filter_data = filter_data.flatten()

    for j in np.arange(1, side_len/2):
        row = j * side_len
        backrow = (side_len - j) * side_len
        row_fact_large = np.exp(-(j*j) * scale_large);
        row_fact_small = np.exp(-(j*j) * scale_small);

        for col in np.arange(1, side_len/2):
            backcol = side_len - col
            col_fact_large = np.exp(-(col*col) * scale_large)
            col_fact_small = np.exp(-(col*col) * scale_small)

            factor = (1 - row_fact_large*col_fact_large) * row_fact_small*col_fact_small

            fft_data[col+row] = fft_data[col+row] * factor
            fft_data[col+backrow] = fft_data[col+backrow] * factor
            fft_data[backcol+row] = fft_data[backcol+row] * factor
            fft_data[backcol+backrow] = fft_data[backcol+backrow] * factor

            filter_data[col+row] = filter_data[col+row] * factor
            filter_data[col+backrow] = filter_data[col+backrow] * factor
            filter_data[backcol+row] = filter_data[backcol+row] * factor
            filter_data[backcol+backrow] = filter_data[backcol+backrow] * factor


    #process meeting points (maxN/2,0) , (0,maxN/2), and (maxN/2,maxN/2)
	rowmid = side_len * (side_len/2)
	row_fact_large = np.exp(- (side_len/2)*(side_len/2) * scale_large)
	row_fact_small = np.exp(- (side_len/2)*(side_len/2) * scale_small)

	fft_data[side_len/2] = fft_data[side_len/2] * (1 - row_fact_large) * row_fact_small
	fft_data[rowmid] = fft_data[rowmid] * (1 - row_fact_large) * row_fact_small
	fft_data[side_len/2 + rowmid] = fft_data[side_len/2 + rowmid] * (1 - row_fact_large * row_fact_large) * row_fact_small * row_fact_small

    filter_data[side_len/2] = filter_data[side_len/2] * (1 - row_fact_large) * row_fact_small
    filter_data[rowmid] = filter_data[rowmid] * (1 - row_fact_large) * row_fact_small
    filter_data[side_len/2 + rowmid] = filter_data[side_len/2 + rowmid] * (1 - row_fact_large * row_fact_large) * row_fact_small * row_fact_small

    #loop along row 0 and side_len/2
    row_fact_large = np.exp(- (side_len/2)*(side_len/2) * scale_large)
    row_fact_small = np.exp(- (side_len/2)*(side_len/2) * scale_small)

    for col in np.arange(1, side_len/2):
        backcol = side_len - col
        col_fact_large = np.exp(- (col*col) * scale_large)
        col_fact_small = np.exp(- (col*col) * scale_small)

        fft_data[col] = fft_data[col] * (1 - col_fact_large) * col_fact_small
        fft_data[backcol] = fft_data[backcol] * (1 - col_fact_large) * col_fact_small
        fft_data[col+rowmid] = fft_data[col+rowmid] * (1 - col_fact_large*row_fact_large) * col_fact_small*row_fact_small
        fft_data[backcol+rowmid] = fft_data[backcol+rowmid] * (1 - col_fact_large*row_fact_large) * col_fact_small*row_fact_small
        filter_data[col] = filter_data[col] * (1 - col_fact_large) * col_fact_small
        filter_data[backcol] = filter_data[backcol] * (1 - col_fact_large) * col_fact_small
        filter_data[col+rowmid] = filter_data[col+rowmid] * (1 - col_fact_large*row_fact_large) * col_fact_small*row_fact_small
        filter_data[backcol+rowmid] = filter_data[backcol+rowmid] * (1 - col_fact_large*row_fact_large) * col_fact_small*row_fact_small

    #loop along column 0 and side_len/2
    col_fact_large = np.exp(- (side_len/2)*(side_len/2) * scale_large)
    col_fact_small = np.exp(- (side_len/2)*(side_len/2) * scale_small)

    for j in np.arange(1, side_len/2):
        row = j * side_len
        backrow = (side_len - j) * side_len

        row_fact_large = np.exp(- (j*j) * scale_large)
        row_fact_small = np.exp(- (j*j) * scale_small)

        fft_data[row] = fft_data[row] * (1 - row_fact_large) * row_fact_small
        fft_data[backrow] = fft_data[backrow] * (1 - row_fact_large) * row_fact_small
        fft_data[row+side_len/2] = fft_data[row+side_len/2] * (1 - row_fact_large*col_fact_large) * row_fact_small*col_fact_small
        fft_data[backrow+side_len/2] = fft_data[backrow+side_len/2] * (1 - row_fact_large*col_fact_large) * row_fact_small*col_fact_small
        filter_data[row] = filter_data[row] * (1 - row_fact_large) * row_fact_small
        filter_data[backrow] = filter_data[backrow] * (1 - row_fact_large) * row_fact_small
        filter_data[row+side_len/2] = filter_data[row+side_len/2] * (1 - row_fact_large*col_fact_large) * row_fact_small*col_fact_small
        filter_data[backrow+side_len/2] = filter_data[backrow+side_len/2] * (1 - row_fact_large*col_fact_large) * row_fact_small*col_fact_small

    fft_data = np.reshape(fft_data, fft_data_shape)
    filter_data = np.reshape(filter_data, fft_data_shape)

    return fft_data, filter_data

def main():
    data_path = r"/Users/rshkarin/Documents/AllaData/afm_data_16bit_512x512.raw"
    data = np.memmap(data_path, dtype=np.int16, shape=(512,512), mode='r')

    height, width = data.shape
    pad_height, pad_width = next_length_pow2(height + 1), next_length_pow2(width + 1)

    padded_data = np.zeros((pad_height, pad_width), dtype=np.int16)
    pad_offset_y, pad_offset_x = pad_height/2 - height/2, pad_width/2 - width/2

    crop_bbox = np.index_exp[pad_offset_y:pad_offset_y + height, \
                             pad_offset_x:pad_offset_x + width]

    padded_data[crop_bbox] = data

    fft_data = np.fft.fft2(padded_data)

    filter_large_dia = 15
    filter_small_dia = 5
    filtered_fft_data, filter_data = filter(fft_data, filter_large_dia, filter_small_dia)

    ifft_data = np.fft.ifft2(filtered_fft_data)
    filtered_data = ifft_data.real[crop_bbox].astype(np.float32)

    p2, p98 = np.percentile(filtered_data, (1, 99))
    filtered_rescaled_data = exposure.rescale_intensity(filtered_data, in_range=(p2, p98))

    filtered_data.tofile(r"/Users/rshkarin/Documents/AllaData/filtered_afm_data_16bit_512x512.raw")
    filtered_rescaled_data.tofile(r"/Users/rshkarin/Documents/AllaData/filtered_rescaled_afm_data_16bit_512x512.raw")

    plt.imshow(filtered_rescaled_data)
    plt.show()


if __name__ == "__main__":
    main()
