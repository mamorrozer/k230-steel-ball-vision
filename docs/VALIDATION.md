# Validation record

## Model comparison

The two full-frame experiments used the same combined dataset, seed, initial YOLO11n model, 100 epochs and batch size 16. On the recorded validation split:

| Input | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| 512 × 288 | 0.9909 | 0.9778 | 0.9830 | 0.8340 |
| 352 × 192 | 0.9744 | 0.9704 | 0.9770 | 0.8170 |

The 352 × 192 candidate gave up a small amount of offline accuracy for a measured K230 local-display throughput increase from about 20 FPS to about 28 FPS. Repeat the K230 measurement if the camera, runtime or lighting changes.

## ROI experiment

The centre ROI 512 × 160 experiment reached mAP50 0.9792 and mAP50-95 0.7867 on its own reviewed ROI split. On the actual K230 scene it produced zero detections and ran at about 12 FPS. The training crop and live camera ROI had different geometry, so this candidate is rejected for the current setup.

## What the measurements do not prove

- They do not prove generalization to a new pipe, board, camera or lighting.
- Offline mAP is not the same as closed-loop control quality.
- Local-LCD FPS is not a guaranteed sensor rate or a 90 FPS result.
- No final pixel-to-millimetre calibration or motor-control closed loop is claimed here.

## Next engineering measurement

For a future control-stage run, use full-frame 352 × 192 first. Record valid detections, misses and false detections; centre jitter with a stationary ball; position error at at least five known physical marks; UART frame interval and invalid-frame handling; and behaviour when the ball leaves the calibrated axis.

