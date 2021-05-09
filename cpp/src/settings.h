#ifndef SETTINGS_H
#define SETTINGS_H

namespace superpoint {
struct Settings {
  bool cuda = false;
  int nms_dist = 4;
  float confidence_thresh = 0.015f;
  float nn_thresh = 0.7f;  // L2 descriptor distance for good match.
  int cell = 8;            // Size of each output cell. Keep this fixed.
  int border_remove = 4;   // Remove points this close to the border.
};
}  // namespace superpoint
#endif  // SETTINGS_H
