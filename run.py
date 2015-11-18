# -*- coding: utf-8 -*-
import platform
import os
import sys
import colorsys
from random import random
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import colormaps as cmaps
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Ellipse, Circle
from scipy import ndimage as ndi
from scipy.misc import imsave
from skimage import exposure
from skimage import filters
from skimage import measure
from skimage.feature import peak_local_max
from skimage.morphology import watershed, disk
from skimage.segmentation import random_walker
from skimage.color import label2rgb

term_enc = 'utf-8' if platform.uname()[0] == 'Linux' else 'cp1251'
matplotlib.style.use('ggplot')

AXES_NAMES = {
    "avg_axis": {"en": "Average axis (nm)", "ru": u"Средняя ось (нм)"},
    "frequency": {"en": "Frequency", 'ru': u"Частота"}
}

def _get_colors(num_colors):
    colors = [(1,1,1)] + [(random(),random(),random()) for i in xrange(255)]
    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('new_cmap', colors, N=len(label_stats.index))

    return new_cmap

def next_length_pow2(x):
    return 2 ** np.ceil(np.log2(abs(x)))

def band_pass_filter(fft_data, filter_large_dia=15, filter_small_dia=5):
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

def particles_stats(segmented_data, properties, min_particle_size=5):
    u_labeled_data = np.unique(segmented_data)
    labeled_data = np.searchsorted(u_labeled_data, segmented_data)

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

def segment_data(data, min_distance=5, footprint=disk(10), indices=False):
    th_val = filters.threshold_otsu(data)
    thresholded_particles = data > th_val
    distance = ndi.distance_transform_edt(thresholded_particles)
    distance = ndi.maximum_filter(distance, footprint=disk(5), mode='nearest')
    local_maxi = peak_local_max(distance, min_distance=min_distance, indices=indices, footprint=footprint, labels=thresholded_particles)
    labeled_data, num_features = ndi.measurements.label(local_maxi)
    segmented_data = watershed(-distance, labeled_data, mask=thresholded_particles)

    return segmented_data, local_maxi

def preprocess_data(data, small_particle=5, large_particle=15):
    height, width = data.shape
    pad_height, pad_width = next_length_pow2(height + 1), next_length_pow2(width + 1)

    padded_data = np.zeros((pad_height, pad_width), dtype=np.int16)
    pad_offset_y, pad_offset_x = pad_height/2 - height/2, pad_width/2 - width/2

    crop_bbox = np.index_exp[pad_offset_y:pad_offset_y + height, \
                             pad_offset_x:pad_offset_x + width]

    padded_data[crop_bbox] = data

    fft_data = np.fft.fft2(padded_data)

    filtered_fft_data, filter_data = band_pass_filter(fft_data, large_particle, small_particle)

    ifft_data = np.fft.ifft2(filtered_fft_data)
    filtered_data = ifft_data.real[crop_bbox].astype(np.float32)

    p2, p98 = np.percentile(filtered_data, (5, 95))
    filtered_rescaled_data = exposure.rescale_intensity(filtered_data, in_range=(p2, p98))

    return filtered_rescaled_data

def create_histogram_figure(stats, output_path, column='avg_axis', range=[], color='r', figsize=(8,6), bins=20, language='en'):
    base_filename='histogram'
    filename_suffix = '.png'

    filtered_data = stats[column]

    if len(range):
       filtered_data = stats[column][(stats[column] >= np.min(range)) & \
                                       (stats[column] <= np.max(range))]
    fig, ax = plt.subplots()
    fig.set_size_inches(figsize)
    counts, bins, patches = ax.hist(filtered_data.values, bins=bins, color=color)
    ax.set_xticks(bins)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%0.f'))
    plt.xlabel(AXES_NAMES[column][language])
    plt.ylabel(AXES_NAMES['frequency'][language])
    plt.savefig(os.path.join(output_path, base_filename + '_' + column + filename_suffix), bbox_inches='tight')
    plt.show()

def create_figures(particles_stats):
    pass

def create_overlay_figure(data, data_mask, label_stats, filename, output_path, base_filename='label_overlay', filename_suffix='.png', figsize=(8,8)):
    if not len(label_stats.index):
        print 'No data stats collected.'
        sys.exit(1)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([.0, .0, 1.0, 1.0])
    ax.imshow(data, cmap=cm.gray, interpolation='bicubic')
    ax.autoscale(False)
    ax.set_adjustable('box-forced')
    ax.set_axis_off()
    ax.imshow(data_mask, alpha=0.3, cmap=_get_colors(len(label_stats.index)), interpolation='bicubic')
    plt.savefig(os.path.join(output_path, '_'.join([filename, base_filename]) + filename_suffix), bbox_inches='tight')

    plt.show()

def create_axis_figure(data, label_stats, filename, output_path, base_filename='axis', file_ext='.png', figsize=(8,8)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([.0, .0, 1.0, 1.0])
    ax.imshow(data, cmap=cm.gray)
    ax.autoscale(False)
    ax.set_adjustable('box-forced')
    ax.set_axis_off()

    for index, row in label_stats.iterrows():
        y0, x0 = row.centroid
        orientation = row.orientation
        x1 = x0 + np.cos(orientation) * 0.5 * row.major_axis_length
        y1 = y0 - np.sin(orientation) * 0.5 * row.major_axis_length
        x2 = x0 - np.sin(orientation) * 0.5 * row.minor_axis_length
        y2 = y0 - np.cos(orientation) * 0.5 * row.minor_axis_length

        ax.plot((x0, x1), (y0, y1), '-r', linewidth=1.5)
        ax.plot((x0, x2), (y0, y2), '-r', linewidth=1.5)
        ax.plot(x0, y0, '.g', markersize=5)

        approx_particle = Circle((x0, y0), radius=row.avg_axis*0.5, edgecolor='b', linewidth=1, fill=False)
        ax.add_patch(approx_particle)

    plt.savefig(os.path.join(output_path, '_'.join([filename, base_filename]) + file_ext), bbox_inches='tight')
    plt.show()

def main():
    root_folder_path = "/Users/rshkarin/Documents/AllaData"
    data_path = os.path.join(root_folder_path, "afm_data_16bit_512x512.raw")
    data = np.memmap(data_path, dtype=np.int16, shape=(512,512), mode='r')
    properties=['label','area','centroid','equivalent_diameter','major_axis_length','minor_axis_length','orientation','bbox']

    processed_data = preprocess_data(data)
    segmented_data, local_maxi = segment_data(processed_data)
    label_stats = particles_stats(segmented_data, properties)
    processed_stats, columns = process_stats(label_stats)

    output_path = root_folder_path

    processed_data.tofile(os.path.join(output_path, "processed_afm_data_16bit_512x512.raw"))
    segmented_data.tofile(os.path.join(output_path, "segmented_filtered_rescaled_afm_data_16bit_512x512.raw"))

    create_overlay_figure(data, segmented_data, label_stats, "sample0001", output_path)
    create_axis_figure(data, label_stats, "sample0001", output_path)
    create_histogram_figure(processed_stats, output_path, range=range(10,400), bins=20, language='en')

if __name__ == "__main__":
    main()
