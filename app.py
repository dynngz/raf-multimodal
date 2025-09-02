import os
import shutil
import gradio as gr
import base64
import io
from PIL import Image
from typing import List, Tuple, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn
import asyncio
from p_m import initialize_rag_system, query_rag_system, rag_system
class QuestionRequest(BaseModel):
    question: str
class ImageData(BaseModel):
    base64: str
    description: str = ""
class RAGResponse(BaseModel):
    text: str
    images: List[ImageData]
    success: bool = True
    message: str = ""
class StatusResponse(BaseModel):
    status: str
    message: str
    document_processed: bool = False
system_initialized = False
current_pdf_path = None
app = FastAPI(title="Multimodal RAG API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
async def root():
    return {
        "message": "Multimodal RAG API esta funcionando",
        "status": "active",
        "version": "1.0.0"
    }
@app.get("/status", response_model=StatusResponse)
async def get_status():
    global system_initialized, current_pdf_path
    return StatusResponse(
        status="ready" if system_initialized else "waiting",
        message="Sistema listo para consultas" if system_initialized else "Esperando procesamiento de documento",
        document_processed=system_initialized
    )
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global system_initialized, current_pdf_path
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Solo se permiten archivos PDF")
    try:
        os.makedirs("./content", exist_ok=True)
        file_path = os.path.join("content", file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        result = initialize_rag_system(file_path)
        if "Error" in result:
            raise HTTPException(status_code=500, detail=result)
        current_pdf_path = file_path
        system_initialized = True
        return {"message": result, "filename": file.filename, "success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando archivo: {str(e)}")
@app.post("/initialize")
async def initialize_system():
    global system_initialized, current_pdf_path
    
    default_pdf_path = "./content/rag-challenge.pdf"
    if not os.path.exists(default_pdf_path):
        raise HTTPException(
            status_code=404,
            detail="No se encontro el archivo PDF por defecto. Sube un archivo usando /upload_pdf"
        )
    try:
        result = initialize_rag_system(default_pdf_path)
        if "Error" in result:
            raise HTTPException(status_code=500, detail=result)
        current_pdf_path = default_pdf_path
        system_initialized = True
        return {"message": result, "success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error inicializando sistema: {str(e)}")

@app.post("/ask", response_model=RAGResponse)
async def ask_question(request: QuestionRequest):
    global system_initialized
    
    if not system_initialized:
        raise HTTPException(
            status_code=400,
            detail="Sistema no inicializado"
        )
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacia")
    try:
        response = query_rag_system(request.question)
        images_data = [
            ImageData(
                base64=img["base64"],
                description=img.get("description", "")
            ) for img in response["images"]
        ]
        return RAGResponse(
            text=response["text"],
            images=images_data,
            success=True,
            message="Consulta procesada exitosamente"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando consulta: {str(e)}")

@app.get("/health")
async def health_check():
    global system_initialized, current_pdf_path
    return {
        "api_status": "healthy",
        "rag_system_initialized": system_initialized,
        "current_document": current_pdf_path if current_pdf_path else None,
        "groq_api_configured": bool(os.getenv("GROQ_API_KEY")),
    }
def check_backend_status():
    global system_initialized
    return system_initialized
def initialize_system_frontend():
    global system_initialized, current_pdf_path
    default_pdf_path = "./content/rag-challenge.pdf"
    if not os.path.exists(default_pdf_path):
        return "No se encontro el archivo pdf", False
    try:
        result = initialize_rag_system(default_pdf_path)
        if "Error" in result:
            return f"Error: {result}", False
        current_pdf_path = default_pdf_path
        system_initialized = True
        return result, True
    except Exception as e:
        return f"Error inicializando sistema: {str(e)}", False
def upload_pdf_frontend(file):
    global system_initialized, current_pdf_path
    if file is None:
        return "Selecciona un archivo PDF", False
    if not file.name.lower().endswith('.pdf'):
        return "Solo se permiten archivos PDF", False
    try:
        os.makedirs("./content", exist_ok=True)
        file_path = os.path.join("content", os.path.basename(file.name))
        shutil.copy2(file.name, file_path)
        result = initialize_rag_system(file_path)
        if "Error" in result:
            return f"Error: {result}", False
        current_pdf_path = file_path
        system_initialized = True
        return result, True
    except Exception as e:
        return f"Error procesando archivo: {str(e)}", False
def base64_to_image(base64_str: str) -> Image.Image:
    try:
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception as e:
        print(f"Error converting base64 to image: {e}")
        return None
def ask_question_frontend(question: str, chat_history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]], Optional[Image.Image], str]:
    global system_initialized
    if not question.strip():
        return "", chat_history, None, ""
    if not system_initialized:
        error_msg = "El sistema no esta iniciado"
        chat_history.append((question, error_msg))
        return "", chat_history, None, ""
    try:
        response = query_rag_system(question)
        response_text = response["text"]
        selected_image = None
        image_description = ""
        if response["images"] and len(response["images"]) > 0:
            img_data = response["images"][0]
            selected_image = base64_to_image(img_data["base64"])
            image_description = img_data.get("description", "Imagen relevante encontrada")
        chat_history.append((question, response_text))
        return "", chat_history, selected_image, image_description
    except Exception as e:
        error_msg = f"Error procesando consulta: {str(e)}"
        chat_history.append((question, error_msg))
        return "", chat_history, None, ""
def get_system_status():
    global system_initialized
    if system_initialized:
        return "Sistema listo para consultas"
    else:
        return "Sistema no inicializado"
def create_interface():
    with gr.Blocks(
        title="RAG Multimodal",
        theme=gr.themes.Soft(),
        css="""
 body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f6f9; }
 .container { max-width: 1400px; margin: auto; padding: 20px; }
 .gr-box { box-shadow: 0 4px 12px rgba(0,0,0,0.05); border-radius: 12px; border: none; background-color: #ffffff; padding: 20px; }
 h1, h3 { color: #333333; }
 .status-box { padding: 12px 20px; margin: 10px 0; border-radius: 8px; font-weight: bold; border-left: 5px solid; }
 .status-ready { background-color: #d4edda; border-left-color: #28a745; color: #155724; }
 .status-initializing { background-color: #cce5ff; border-left-color: #007bff; color: #004085; }
 .status-error { background-color: #f8d7da; border-left-color: #dc3545; color: #721c24; }
 .status-not-initialized { background-color: #fff3cd; border-left-color: #ffc107; color: #856404; }
 .chat-container { height: 400px; overflow-y: auto; background-color: #f9f9f9; border-radius: 8px; border: 1px solid #e0e0e0; }
 .gr-button.primary { background-color: #6c5ce7; color: white; border: none; border-radius: 8px; }
 .gr-button.secondary { background-color: #e9ecef; color: #495057; border: none; border-radius: 8px; }
 .gr-button.success { background-color: #28a745; color: white; border: none; border-radius: 8px; }
 .upload-section { background-color: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #e9ecef; }
 .image-description { font-style: italic; color: #666; margin-top: 10px; padding: 8px; background-color: #f0f0f0; border-radius: 5px; }
 """
    ) as demo:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("# RAG Multimodal")
                gr.Markdown("### Estado del sistema")
                status_html = gr.HTML(
                    value="""<div class=\"status-box status-not-initialized\">Sistema no iniciado. Presiona 'Inicializar Sistema' para comenzar.</div>"""
                )
                initialize_btn = gr.Button("Inicializar Sistema", variant="primary")
                gr.Markdown("### Cambiar documento")
                pdf_upload = gr.File(
                    label="Seleccionar nuevo archivo PDF",
                    file_types=[".pdf"],
                    file_count="single"
                )
                upload_btn = gr.Button("Procesar Nuevo PDF", variant="secondary")
                upload_output = gr.Textbox(
                    label="Estado de Procesamiento",
                    interactive=False,
                    lines=3,
                    visible=False
                )
                gr.Markdown("### Preguntas de ejemplo")
                example_1 = gr.Button("Processing Flow y Human-in-the-loop", variant="secondary")
                example_2 = gr.Button("Fecha de nacimiento del carnet", variant="secondary")
                example_3 = gr.Button("Arquitectura del sistema", variant="secondary")
            with gr.Column(scale=2):
                gr.Markdown("### Chat")
                chatbot = gr.Chatbot(
                    label="Conversacion",
                    height=400,
                    placeholder="Presiona 'Iniciar Sistema' para comenzar"
                )
                with gr.Row():
                    question_input = gr.Textbox(
                        label="Tu pregunta",
                        lines=2,
                        scale=4,
                        interactive=False
                    )
                    ask_btn = gr.Button("Enviar", variant="primary", scale=1, interactive=False)
                clear_btn = gr.Button("Limpiar chat", variant="secondary")
                gr.Markdown("### Imagen")
                selected_image = gr.Image(
                    label="Imagen",
                    show_label=False,
                    type="pil",
                    interactive=False,
                    height=300
                )
                image_description = gr.Textbox(
                    label="Descripcion de la Imagen",
                    interactive=False,
                    lines=2,
                    placeholder="Aqui aparecera la descripcion de la imagen seleccionada..."
                )
        def handle_initialization():
            result, success = initialize_system_frontend()
            status_html_value = f"""<div class=\"status-box status-ready\">Sistema listo para consultas</div>""" if success else f"""<div class=\"status-box status-error\">Error al iniciar el sistema</div>"""
            return (
                status_html_value,
                gr.update(interactive=True, placeholder="Escribe tu pregunta sobre el documento..."),
                gr.update(interactive=True),
                gr.update(placeholder="Conversacion")
            )
        def handle_upload(file):
            if file is None:
                return gr.update(value="No se selecciono ningun archivo", visible=True), gr.update(), gr.update(interactive=False), gr.update(interactive=False), gr.update()
            result, success = upload_pdf_frontend(file)
            status_html_value = f"""<div class=\"status-box status-ready\">Sistema listo para consultas</div>""" if success else f"""<div class=\"status-box status-error\">Error al procesar el documento</div>"""
            return (
                gr.update(value=result, visible=True),
                status_html_value,
                gr.update(interactive=True, placeholder="Escribe tu pregunta sobre el documento..."),
                gr.update(interactive=True),
                gr.update(placeholder="Conversacion")
            )
        def handle_question(question, history):
            if not question.strip():
                return "", history, None, ""
            return ask_question_frontend(question, history)
        def clear_chat():
            return [], None, ""
        initialize_btn.click(
            handle_initialization,
            outputs=[status_html, question_input, ask_btn, chatbot]
        )
        upload_btn.click(
            handle_upload,
            inputs=[pdf_upload],
            outputs=[upload_output, status_html, question_input, ask_btn, chatbot]
        )
        ask_btn.click(
            handle_question,
            inputs=[question_input, chatbot],
            outputs=[question_input, chatbot, selected_image, image_description]
        )
        question_input.submit(
            handle_question,
            inputs=[question_input, chatbot],
            outputs=[question_input, chatbot, selected_image, image_description]
        )
        clear_btn.click(
            clear_chat,
            outputs=[chatbot, selected_image, image_description]
        )
        example_1.click(lambda: "Que pasos del Processing Flow se asocian con la funcionalidad de Human-in-the-loop review?", outputs=[question_input])
        example_2.click(lambda: "Cual es la fecha de nacimiento que se extrae del carnet en la interfaz de usuario?", outputs=[question_input])
        example_3.click(lambda: "Explicame la arquitectura del sistema", outputs=[question_input])
    return demo
demo = create_interface()
app = gr.mount_gradio_app(app, demo, path="/")
if __name__ == "__main__":
    os.makedirs("./content", exist_ok=True)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 7860)),
        log_level="info"
    )