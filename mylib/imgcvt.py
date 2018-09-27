import collections
import os
import typing

import cv2 as cv

ImageConvert = collections.namedtuple(
    "ImageConvert", "process_name process imread_color")
ImageConvert.IMREAD_GRAYSCALE = 0
ImageConvert.IMREAD_COLOR = 1


class ImageConvertRunner:
    output_folder_path = "output/"

    def __init__(self,
                 original_file_path: str,
                 converts: typing.Sequence[ImageConvert]):
        self.original_file_path = original_file_path
        self.converts = converts

    def run(self, index: int):
        convert = self.converts[index]

        input_file_path = self._input_file_path(index)
        if convert.imread_color == ImageConvert.IMREAD_COLOR:
            input_mode = cv.IMREAD_COLOR
        else:
            input_mode = cv.IMREAD_GRAYSCALE
        input = cv.imread(input_file_path, input_mode)

        output = convert.process(input)
        output_file_path = self._output_file_path(index)
        cv.imwrite(output_file_path, output)

    def run_all(self):
        for i, _ in enumerate(self.converts):
            self.run(i)

    def _output_file_path(self, index: int):
        max_index_len = len(self.converts) // 10 + 1
        index_str = f"{index:0{max_index_len}}"
        convert = self.converts[index]
        output_file_name = f"{index_str}_{convert.process_name}.jpg"
        return os.path.join(self.output_folder_path, output_file_name)

    def _input_file_path(self, index: int):
        # HACK: When both `_input_file_path` method and `_output_file_path`
        # one are calld, they do same process. Caching is needed.
        if index == 0:
            return self.original_file_path
        else:
            return self._output_file_path(index - 1)
