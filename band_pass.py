import numpy as np
import matplotlib.pyplot as plt

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

            fft_data[]

	for (int j=1; j<maxN/2; j++) {
			row = j * maxN;
			backrow = (maxN-j)*maxN;
			rowFactLarge = (float) Math.exp(-(j*j) * scaleLarge);
			rowFactSmall = (float) Math.exp(-(j*j) * scaleSmall);


			// loop over columns
			for (col=1; col<maxN/2; col++){
				backcol = maxN-col;
				colFactLarge = (float) Math.exp(- (col*col) * scaleLarge);
				colFactSmall = (float) Math.exp(- (col*col) * scaleSmall);
				factor = (1 - rowFactLarge*colFactLarge) * rowFactSmall*colFactSmall;
				switch (stripesHorVert) {
					case 1: factor *= (1 - (float) Math.exp(- (col*col) * scaleStripes)); break;// hor stripes
					case 2: factor *= (1 - (float) Math.exp(- (j*j) * scaleStripes)); // vert stripes
				}

				fht[col+row] *= factor;
				fht[col+backrow] *= factor;
				fht[backcol+row] *= factor;
				fht[backcol+backrow] *= factor;
				filter[col+row] *= factor;
				filter[col+backrow] *= factor;
				filter[backcol+row] *= factor;
				filter[backcol+backrow] *= factor;
			}
		}




def main():
    data_path = "E:\\fiji-win64\\AllaData\\data_16bit_512x512.raw"
    data = np.memmap(data_path, dtype=np.int16, shape=(512,512), mode='r')

    height, width = data.shape
    pad_height, pad_width = next_length_pow2(height + 1), next_length_pow2(width + 1)

    padded_data = np.zeros((pad_height, pad_width), dtype=np.int16)
    pad_offset_y, pad_offset_x = pad_height/2 - height/2, pad_width/2 - width/2

    padded_data[pad_offset_y:pad_offset_y + height, \
                pad_offset_x:pad_offset_x + height] = data

    fft_data = np.fft.fft2(padded_data)
    fft_data = np.fft.fftshift(fft_data)



    plt.imshow(padded_data)
    plt.show()


if __name__ == "__main__":
    main()
