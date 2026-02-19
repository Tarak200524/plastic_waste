import os

BASE_DIR = os.getcwd()

def rename_split(split):
    img_dir = os.path.join(BASE_DIR, "images", split)
    lbl_dir = os.path.join(BASE_DIR, "labels", split)

    images = sorted([f for f in os.listdir(img_dir)
                     if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))])

    labels = sorted([f for f in os.listdir(lbl_dir) if f.endswith(".txt")])

    image_names = set(os.path.splitext(f)[0] for f in images)
    label_names = set(os.path.splitext(f)[0] for f in labels)

    common = sorted(image_names.intersection(label_names))

    print(f"\nProcessing {split}")
    print(f"Matched pairs: {len(common)}")

    counter = 1

    for name in common:
        # detect image extension
        for ext in [".jpg", ".jpeg", ".png", ".webp"]:
            img_path = os.path.join(img_dir, name + ext)
            if os.path.exists(img_path):
                image_ext = ext
                break

        old_img = os.path.join(img_dir, name + image_ext)
        old_lbl = os.path.join(lbl_dir, name + ".txt")

        new_name = f"{split}_{counter:04d}"

        new_img = os.path.join(img_dir, new_name + image_ext)
        new_lbl = os.path.join(lbl_dir, new_name + ".txt")

        os.rename(old_img, new_img)
        os.rename(old_lbl, new_lbl)

        counter += 1

    print(f"{split} renamed successfully.")

if __name__ == "__main__":
    rename_split("train")
    rename_split("val")
    print("\nAll files renamed cleanly.")
