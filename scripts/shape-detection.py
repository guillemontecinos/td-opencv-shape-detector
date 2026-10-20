import numpy as np
import cv2

def onCook(scriptOp):
    # 1. Obtener la imagen de entrada
    img_top = scriptOp.inputs[0].numpyArray(delayed=True)
    if img_top is None:
        return
        
    # 2. Convertir el formato float32 de TouchDesigner a uint8 para OpenCV
    # Ignoramos el canal Alpha ([: , :, :3]) porque OpenCV trabaja mejor en BGR
    img_8u = (img_top[:, :, :3] * 255).astype(np.uint8)
    
    # 3. Convertir a escala de grises
    gray = cv2.cvtColor(img_8u, cv2.COLOR_BGR2GRAY)
    
    # 4. Umbral INVERTIDO: Convierte el fondo blanco en negro y las líneas negras en blancas
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    
    # 5. Encontrar los contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 6. Preparar la tabla de salida (DAT)
    dat = op('shape_data')
    dat.clear()
    dat.appendRow(['ID', 'Forma', 'Vertices', 'Centro_X', 'Centro_Y'])
    
    # Hacemos una copia de la imagen para dibujar los resultados encima
    output_img = img_8u.copy()
    
    shape_id = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Ignorar ruido muy pequeño o contornos gigantes (como el borde del archivo completo)
        if area < 50 or area > 500000:
            continue
            
        # Calcular el perímetro
        peri = cv2.arcLength(cnt, True)
        
        # 7. Aproximación de Polígonos (EL CORAZÓN DEL SISTEMA)
        # Epsilon determina la sensibilidad. 4% del perímetro suele ser ideal para formas rectas.
        epsilon = 0.04 * peri
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        # Contar cuántos vértices tiene la figura aproximada
        vertices = len(approx)
        
        # 8. Clasificar según los vértices
        if vertices == 3:
            shape_name = "Triangulo"
        elif vertices == 4:
            # Puedes usar el aspect ratio para separar rectángulos de cuadrados
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w)/h
            if 0.95 <= aspect_ratio <= 1.05:
                shape_name = "Cuadrado"
            else:
                shape_name = "Rectangulo"
        elif vertices == 5:
            shape_name = "Pentagono"
        elif vertices > 8:
            shape_name = "Circulo/Curva"
        else:
            shape_name = f"Custom ({vertices})"
            
        # 9. Encontrar el centro matemático (Momentos) de la forma
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
        else:
            cx, cy = 0, 0
            
        # Dibujar los contornos en verde y el texto en azul
        cv2.drawContours(output_img, [approx], -1, (0, 255, 0), 3)
        cv2.putText(output_img, shape_name, (cx - 40, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Guardar los datos en el Table DAT
        # Normalizamos X y Y (0 a 1) para que sea fácil usar en TouchDesigner
        height, width = gray.shape
        norm_x = round(cx / width, 3)
        # OpenCV cuenta Y desde arriba, TouchDesigner desde abajo, así que invertimos Y
        norm_y = round(1.0 - (cy / height), 3) 
        
        dat.appendRow([shape_id, shape_name, vertices, norm_x, norm_y])
        shape_id += 1
        
    # 10. Devolver la imagen modificada a TouchDesigner
    output_img_rgba = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGBA)
    out_float = (output_img_rgba / 255.0).astype(np.float32)
    scriptOp.copyNumpyArray(out_float)