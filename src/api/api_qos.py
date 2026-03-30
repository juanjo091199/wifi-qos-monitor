from flask import Flask, jsonify
import pandas as pd
import os

app = Flask(__name__)

CSV_PATH = "/home/juan/dataset_qos.csv"
N_FILAS  = 40  # últimas N muestras para construir ventana temporal

@app.route("/datos", methods=["GET"])
def obtener_datos():
    try:
        if not os.path.exists(CSV_PATH):
            return jsonify({"error": "CSV no encontrado"}), 404
        
        df = pd.read_csv(CSV_PATH)
        ultimas = df.tail(N_FILAS).to_dict(orient="records")
        return jsonify({
            "status": "ok",
            "n_filas": len(ultimas),
            "datos": ultimas
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/estado", methods=["GET"])
def estado():
    return jsonify({"status": "ok", "mensaje": "API QoS activa"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)