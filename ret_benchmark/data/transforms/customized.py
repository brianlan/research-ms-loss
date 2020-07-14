from PIL import Image


def pad_shorter(x):
    h, w = x.size[-2:]
    s = max(h, w)
    new_im = Image.new("RGB", (s, s))
    new_im.paste(x, ((s - h) // 2, (s - w) // 2))
    return new_im


class PadShorter:
    def __call__(self, img):
        return pad_shorter(img)
