from face_filters import FaceAccessory, FaceFilterApp

def main():
    glasses = FaceAccessory("images/gg.png", left_idx=33, right_idx=263, width_scale=1.5)
    short_hair = FaceAccessory("images/short_hair.png", left_idx=103, right_idx=332, width_scale=2.5, vertical_offset_ratio=0.02, horizontal_offset_ratio=0.008)
    long_hair = FaceAccessory("images/long_hair.png", left_idx=103, right_idx=332, width_scale=4.5, vertical_offset_ratio=0.33, horizontal_offset_ratio=0.12)
    beard = FaceAccessory("images/beards.png", left_idx=172, right_idx=397,
                           width_scale=1.4, vertical_offset_ratio=0.05)

    app = FaceFilterApp(accessories=[short_hair, glasses, beard]) #make your favorite combination with this btw 
    app.run()

if __name__ == "__main__":
    main()