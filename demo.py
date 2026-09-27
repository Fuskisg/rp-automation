from vision import ColorSignal, ColorSignalDetector, TemplateMatcher
import numpy as np

def main():
    template = np.zeros((24, 24, 3), dtype=np.uint8)
    template[4:20, 4:20] = (255, 255, 255)
    frame = np.zeros((120, 180, 3), dtype=np.uint8)
    frame[50:74, 80:104] = template
    matcher = TemplateMatcher(threshold=0.9)
    matcher.add('prompt', template)
    print('template:', matcher.match_best(frame))
    red_frame = np.zeros((120, 180, 3), dtype=np.uint8)
    red_frame[30:60, 40:90] = (0, 0, 255)
    detector = ColorSignalDetector([ColorSignal('red', (255, 0, 0), minimum_area=20)])
    print('color:', detector.detect(red_frame))

if __name__ == '__main__':
    main()
