import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import colormaps as cmaps
from scipy import ndimage as ndi
from skimage import exposure
from skimage import filters
from skimage import measure
from skimage.feature import peak_local_max
from skimage.morphology import watershed, disk
from skimage.segmentation import random_walker
from skimage.color import label2rgb

def next_length_pow2(x):
    return 2 ** np.ceil(np.log2(abs(x)))

def band_pass_filter(fft_data, filter_large_dia, filter_small_dia):
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

def particles_stats(segmented_data, properties, min_particle_size=20):
    labeled_data, num_labels = ndi.measurements.label(segmented_data)

    label_sizes = ndi.measurements.sum(segmented_data, labeled_data, np.arange(num_labels + 1))
    filtered_small_particles = label_sizes < min_particle_size
    remove_particles = filtered_small_particles[labeled_data]
    labeled_data[remove_particles] = 0

    u_labeled_data = np.unique(labeled_data)
    labeled_data = np.searchsorted(u_labeled_data, labeled_data)

    stats = pd.DataFrame(columns=properties)

    for region in measure.regionprops(labeled_data):
        stats = stats.append({_property: region[_property] for _property in properties}, \
                                ignore_index=True)

    return stats

def process_stats(particles_stats, pixel_scale_factor=0.512):
    if 'major_axis_length' in particles_stats and 'minor_axis_length' in particles_stats:
        particles_stats['avg_axis'] = (particles_stats['major_axis_length'] + \
                                            particles_stats['minor_axis_length']) / 2.0

    stats_columns = list(particles_stats.columns.values)

    if 'label' in stats_columns:
        stats_columns.remove('label')

    def scale_values(item):
        if isinstance(item,tuple):
            return tuple(x / pixel_scale_factor for x in item)
        else:
            return item / pixel_scale_factor

    particles_stats_scaled = particles_stats.copy()
    particles_stats_scaled[stats_columns] = particles_stats_scaled[stats_columns].applymap(scale_values)

    return particles_stats_scaled, particles_stats_scaled.columns.values

def segment_data(data):
    th_val = filters.threshold_otsu(data)
    thresholded_particles = data > th_val
    distance = ndi.distance_transform_edt(thresholded_particles)
    local_maxi = peak_local_max(distance, min_distance=5, indices=False, footprint=disk(10), labels=thresholded_particles)
    local_maxi = ndi.morphology.binary_dilation(local_maxi, structure=ndi.generate_binary_structure(2, 1), iterations=2)
    labeled_data, num_features = ndi.measurements.label(local_maxi)
    segmented_data = watershed(-distance, labeled_data, mask=thresholded_particles)

    return segmented_data, local_maxi

def preprocess_data(data):
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
    filtered_fft_data, filter_data = band_pass_filter(fft_data, filter_large_dia, filter_small_dia)

    ifft_data = np.fft.ifft2(filtered_fft_data)
    filtered_data = ifft_data.real[crop_bbox].astype(np.float32)

    p2, p98 = np.percentile(filtered_data, (5, 95))
    filtered_rescaled_data = exposure.rescale_intensity(filtered_data, in_range=(p2, p98))
    #filtered_rescaled_data = ndi.filters.gaussian_filter(filtered_rescaled_data, sigma=3.0)

    return filtered_rescaled_data

def create_histogram_figure(stats, output_folder, column='avg_axis', range=None, color='r', figsize=(5,5), bins=20):
    base_filename='histogram'
    filename_suffix = '.svg'

    plt.figure()
    if not range:
        stats[column].plot(kind='hist', bins=bins, color=color, figsize=figsize)
    else:
        stats[column][stats[column].in(range)].plot(kind='hist', bins=bins, color=color, figsize=figsize)
    #plt.hist(particles_stats[col].values, bins=bins, range=range)
    #plt.title(col)
    #plt.xlabel('Particle %s' % col)
    #plt.ylabel('Frequency')
    #plt.x_lim([0, particles_stats[col].mean()])
    #plt.savefig(os.path.join(output_folder, base_filename + '_' + col + filename_suffix))
    plt.show()

def create_figures(particles_stats):
    pass

def main():
    data_path = "E:\\fiji-win64\\AllaData\\data_16bit_512x512.raw"
    data = np.memmap(data_path, dtype=np.int16, shape=(512,512), mode='r')
    properties=['label','area','centroid','equivalent_diameter','major_axis_length','minor_axis_length','orientation']

    processed_data = preprocess_data(data)
    segmented_data, local_maxi = segment_data(processed_data)
    label_stats = particles_stats(segmented_data, properties)
    processed_stats, columns = process_stats(label_stats)

    processed_data.tofile("E:\\fiji-win64\\AllaData\\processed_afm_data_16bit_512x512.raw")
    segmented_data.tofile("E:\\fiji-win64\\AllaData\\segmented_filtered_rescaled_afm_data_16bit_512x512.raw")

    create_histogram_figure(processed_stats, "E:\\fiji-win64\\AllaData", range=np.arange(300))

    # fig, axes = plt.subplots(ncols=3, figsize=(20, 10), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
    # ax0, ax1, ax2 = axes
    #
    # ax0.imshow(processed_data, interpolation='bicubic')
    # ax0.set_title('Preprocessed data')
    #
    # ax1.imshow(segmented_data, cmap='gray')
    # ax1.set_title('Original data')
    #
    # ax2.imshow(label2rgb(local_maxi, image=processed_data))
    # #ax2.imshow(local_maxi)
    # ax2.set_title('Segmented data')
    #
    # for ax in axes:
    #     ax.axis('off')
    #
    # fig.subplots_adjust(hspace=0.01, wspace=0.01, top=0.9, bottom=0, left=0, right=1)
    # plt.show()

if __name__ == "__main__":
    main()
