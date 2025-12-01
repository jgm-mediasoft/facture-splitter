from ultralytics import YOLO

def main():
    # On part d'un modèle pré-entraîné léger
    model = YOLO("yolov8n.pt")

    model.train(
    	data="tampons.yaml",
    	epochs=50,
    	imgsz=640,
    	batch=8,
    	patience=10,
    	device=0,        # 👈 force l’utilisation du GPU 0
    	verbose=True,
    )

    print("\n✅ Entraînement terminé. Modèle dans runs_tampon/yolov8n_tampon/weights/best.pt")


if __name__ == "__main__":
    main()
