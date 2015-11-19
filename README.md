# afm-particle-analysis
The particle analysis for Actomic Force Microscope images

### Example of config file (e.g. param.yaml):
    paths: &PATHS
        input_path: '/input/path'
        output_path: '/output/path/result'

    samples:
      - filename: 'data.000'
        name: 'Sample_001'
        pixel_scale: 0.512
        segmentation:
          min_dist: 5
          peak_footprint: 10
          filter_footprint: 5
        intensity_scale:
          min: 5
          max: 95
        bp_filter:
          min: 5
          max: 15
        histogram:
          lang: 'en'
          range: [0,400]
          bins: 30
          color: 'red'
          figsize: [10,6]
        <<: *PATHS
