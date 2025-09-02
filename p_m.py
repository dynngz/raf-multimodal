import os
from dotenv import load_dotenv
import uuid
import base64
import io
import re
from typing import List, Tuple, Dict, Any
from PIL import Image
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel

from groq import Groq
from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from unstructured.partition.pdf import partition_pdf
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

_processor = None
_clip_model = None

def get_clip_models():
    global _processor, _clip_model
    if _processor is None or _clip_model is None:
        try:
            print("Cargando modelos CLIP...")
            _processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            print("Modelos clip cargados exitosamente")
        except Exception as e:
            print(f"Error cargando modelos clip: {e}")
            raise e
    return _processor, _clip_model

class CLIPEmbeddingsWrapper:
    def __init__(self, processor=None, model=None, max_tokens=77):
        if processor is None or model is None:
            processor, model = get_clip_models()
        self.processor = processor
        self.model = model
        self.max_tokens = max_tokens

    def _truncate_text(self, text: str) -> str:
        truncated_inputs = self.processor(
            text=[text], 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=self.max_tokens
        )
        
        truncated_tokens = truncated_inputs['input_ids'][0]
        truncated_tokens = truncated_tokens[truncated_tokens != self.processor.tokenizer.pad_token_id]
        truncated_text = self.processor.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        
        return truncated_text

    def _chunk_long_text(self, text: str, overlap_tokens: int = 10) -> List[str]:
        words = text.split()
        chunks = []
        
        words_per_chunk = int((self.max_tokens - 2) * 0.75)
        overlap_words = int(overlap_tokens * 0.75)
        
        start_idx = 0
        while start_idx < len(words):
            end_idx = min(start_idx + words_per_chunk, len(words))
            chunk_words = words[start_idx:end_idx]
            chunk_text = " ".join(chunk_words)
            
            chunk_text = self._truncate_text(chunk_text)
            chunks.append(chunk_text)
            
            start_idx = end_idx - overlap_words
            if start_idx >= len(words):
                break
                
        return chunks

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []
        
        for text in texts:
            truncated_text = self._truncate_text(text)
            
            inputs = self.processor(
                text=[truncated_text], 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=self.max_tokens
            )
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
            all_embeddings.append(text_features[0].detach().numpy().tolist())
        
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        truncated_text = self._truncate_text(text)
        inputs = self.processor(text=[truncated_text], return_tensors="pt", padding=True, truncation=True, max_length=self.max_tokens)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        return text_features[0].detach().numpy().tolist()

class MultimodalRAG:
    def __init__(self, output_path: str = "./content/"):
        self.output_path = output_path
        self.texts = []
        self.tables = []
        self.images = []
        self.retriever = None
        self.rag_chain = None

    def process_pdf(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            return f"Error: El archivo {file_path} no existe."
        
        try:
            chunks = partition_pdf(
                filename=file_path,
                infer_table_structure=True,
                strategy="hi_res",
                extract_image_block_types=["Image"],
                extract_image_block_to_payload=True,
                chunking_strategy="by_title",
                max_characters=10000,
                combine_text_under_n_chars=2000,
                new_after_n_chars=6000,
            )

            for chunk in chunks:
                if "Table" in str(type(chunk)):
                    self.tables.append(chunk)
                elif "CompositeElement" in str(type(chunk)):
                    self.texts.append(chunk)
                    chunk_els = chunk.metadata.orig_elements
                    for el in chunk_els:
                        if "Image" in str(type(el)):
                            self.images.append(el)
            
            self.retriever = self._build_retriever()
            self.rag_chain = self._get_rag_chain()
            
            return f"""Documento procesado exitosamente :)
            
Texto: {len(self.texts)} chunks encontrados
Tablas: {len(self.tables)} tablas encontradas
Imagenes: {len(self.images)} imagenes encontradas
            
El sistema esta listo para responder preguntas"""
        except Exception as e:
            return f"Error procesando el documento: {str(e)}"

    def _get_multimodal_description_chain(self):
        messages_multimodal = [
            (
                "user",
                [
                    {"type": "text", "text": "Describe la imagen en detalle. Para contexto, la imagen es parte de un documento tecnico que explica una arquitectura."},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,{image}"},
                    },
                ],
            )
        ]
        prompt_multimodal = ChatPromptTemplate.from_messages(messages_multimodal)
        model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct") 
        chain_multimodal = prompt_multimodal | model | StrOutputParser()
        return chain_multimodal

    def _build_retriever(self):
        processor, clip_model = get_clip_models()
        clip_embeddings = CLIPEmbeddingsWrapper(processor, clip_model)

        vectorstore = Chroma(
            collection_name="multi_modal_rag", 
            embedding_function=clip_embeddings
        )
        
        store = InMemoryStore()
        id_key = "doc_id"
        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            docstore=store,
            id_key=id_key,
        )

        if self.texts:
            doc_ids_text = [str(uuid.uuid4()) for _ in self.texts]
            docs_to_add = [
                Document(
                    page_content=text.text, 
                    metadata={
                        id_key: doc_ids_text[i],
                        "type": "text",
                        "page_number": getattr(text.metadata, 'page_number', None)
                    }
                ) for i, text in enumerate(self.texts)
            ]
            retriever.vectorstore.add_documents(docs_to_add)
            retriever.docstore.mset(list(zip(doc_ids_text, [{"content": t.text, "type": "text"} for t in self.texts])))
            
        if self.tables:
            doc_ids_tables = [str(uuid.uuid4()) for _ in self.tables]
            docs_to_add = [
                Document(
                    page_content=table.text_as_html, 
                    metadata={
                        id_key: doc_ids_tables[i],
                        "type": "table",
                        "page_number": getattr(table.metadata, 'page_number', None)
                    }
                ) for i, table in enumerate(self.tables)
            ]
            retriever.vectorstore.add_documents(docs_to_add)
            retriever.docstore.mset(list(zip(doc_ids_tables, [{"content": t.text, "type": "table"} for t in self.tables])))
            
        if self.images:
            doc_ids_images = [str(uuid.uuid4()) for _ in self.images]
            
            multimodal_description_chain = self._get_multimodal_description_chain()
            image_summaries = multimodal_description_chain.batch(
                [img.metadata.image_base64 for img in self.images], {"max_concurrency": 3}
            )
            
            summary_docs = []
            for i, summary in enumerate(image_summaries):
                caption = getattr(self.images[i].metadata, 'caption', '')
                text_context = getattr(self.images[i].metadata, 'text_as_html', '')
                page_number = getattr(self.images[i].metadata, 'page_number', None)
                
                enriched_summary = f"{summary}\n\nDescripcion del documento: {caption}\n{text_context}"
                
                summary_docs.append(
                    Document(
                        page_content=enriched_summary, 
                        metadata={
                            id_key: doc_ids_images[i],
                            "type": "image",
                            "page_number": page_number,
                            "image_index": i  
                        }
                    )
                )
            
            retriever.vectorstore.add_documents(summary_docs)
            retriever.docstore.mset(list(zip(doc_ids_images, [
                {
                    "content": img.metadata.image_base64, 
                    "type": "image",
                    "description": image_summaries[i],
                    "image_index": i
                } for i, img in enumerate(self.images)
            ])))
        
        return retriever

    def _parse_docs_enhanced(self, docs):
        images_b64 = []
        texts = []
        
        for doc in docs:
            if isinstance(doc, dict):
                if doc.get("type") == "image":
                    images_b64.append({
                        "base64": doc["content"],
                        "description": doc.get("description", ""),
                        "image_index": doc.get("image_index", 0)
                    })
                else:
                    texts.append(doc["content"])
            else:
                try:
                    base64.b64decode(doc)
                    images_b64.append({"base64": doc, "description": "", "image_index": 0})
                except Exception:
                    texts.append(doc if isinstance(doc, str) else str(doc))
        
        return {
            "images": images_b64,
            "texts": texts
        }

    def _build_prompt_enhanced(self, kwargs):
        docs_by_type = kwargs["context"]
        user_question = kwargs["question"]

        context_text = ""
        if len(docs_by_type["texts"]) > 0:
            for text_element in docs_by_type["texts"]:
                context_text += text_element + "\n"

        image_context = ""
        if len(docs_by_type["images"]) > 0:
            image_context = "\n\nimagenes disponibles para referencia:\n"
            for i, img_data in enumerate(docs_by_type["images"]):
                if img_data.get("description"):
                    image_context += f"IMAGEN_{i+1}: {img_data['description']}\n"

        prompt_template = f"""
        Responde a la pregunta basandote unicamente en el siguiente contexto, que puede incluir texto, tablas y las imagenes que se muestran a continuacion.

        Contexto textual: {context_text}
        {image_context}

        Pregunta: {user_question}
        
        INSTRUCCIONES CRITICAS:
        1. DEBES proporcionar una respuesta COMPLETA y DETALLADA. NO cortes la respuesta abruptamente.
        2. Si la pregunta involucra fechas, numeros de version o calculos, se muy especifico y muestra todos los pasos.
        3. Para preguntas sobre diferencias entre fechas, calcula explicitamente los dias de diferencia.
        4. Si encuentras discrepancias en fechas o informacion, explica TODAS las diferencias encontradas.
        5. Si hay una imagen especifica que ilustra mejor tu respuesta, incluye EXACTAMENTE el marcador [IMAGEN_X] donde X es el numero de la imagen.
        6. Solo incluye UN marcador de imagen por respuesta, el mas relevante.
        7. IMPORTANTE: Completa siempre tu analisis antes de terminar la respuesta.
        8. Si la pregunta tiene multiples partes, responde a TODAS las partes, sigue un chain-of-thought.
        """

        prompt_content = [{"type": "text", "text": prompt_template}]

        if len(docs_by_type["images"]) > 0:
            for img_data in docs_by_type["images"]:
                prompt_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_data['base64']}"},
                    }
                )

        return ChatPromptTemplate.from_messages([HumanMessage(content=prompt_content)])

    def _extract_image_marker(self, response_text: str) -> Tuple[str, int]:

        pattern = r'\[IMAGEN_(\d+)\]'
        matches = re.findall(pattern, response_text)
        
        if matches:
            image_number = int(matches[-1])
            cleaned_text = re.sub(pattern, '', response_text).strip()
            return cleaned_text, image_number - 1  
        
        return response_text, -1

    def _get_rag_chain(self):
        llm = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.2, 
            max_tokens=4096,  
            top_p=0.95,
            frequency_penalty=0.1,  
            stop=None  
        )
        
        chain = (
            {
                "context": self.retriever | RunnableLambda(self._parse_docs_enhanced),
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(self._build_prompt_enhanced)
            | llm
            | StrOutputParser()
        )
        return chain

    def ask_question(self, question: str) -> Dict[str, Any]:
        if not self.rag_chain:
            return {
                "text": "El sistema RAG no esta iniciado",
                "images": []
            }
        
        try:
            retrieved_docs = self.retriever.get_relevant_documents(question)
            parsed_docs = self._parse_docs_enhanced(retrieved_docs)
            
            response_text = self.rag_chain.invoke(question)
            
            cleaned_text, selected_image_index = self._extract_image_marker(response_text)
            
            selected_images = []
            if selected_image_index >= 0 and selected_image_index < len(parsed_docs["images"]):
                selected_image = parsed_docs["images"][selected_image_index]
                selected_images.append({
                    "base64": selected_image["base64"],
                    "description": selected_image.get("description", "")
                })
            
            return {
                "text": cleaned_text,
                "images": selected_images
            }
            
        except Exception as e:
            return {
                "text": f"Error al procesar la pregunta: {str(e)}",
                "images": []
            }

rag_system = MultimodalRAG()

def initialize_rag_system(pdf_path: str = "../content/rag-challenge.pdf") -> str:
    global rag_system
    return rag_system.process_pdf(pdf_path)

def query_rag_system(question: str) -> Dict[str, Any]:
    global rag_system
    return rag_system.ask_question(question)

if __name__ == "__main__":
    pdf_path = "../content/rag-challenge.pdf"
    
    result = initialize_rag_system(pdf_path)
    print(result)
    
    if rag_system.rag_chain:
        test_questions = [
            "¿Qué pasos del \"Processing Flow\" que se muestra en la captura de pantalla de la interfaz de usuario se asocian con la funcionalidad de \"Human-in-the-loop (HITL) review\" descrita en el texto, y qué servicio de AWS se menciona para esta funcionalidad en particular?",
            "¿cuál es la fecha de nacimiento que se extrae del carnet en la interfaz de usuario de \"Visual Document Editor\"?",
            "Explícame la arquitectura del sistema"
        ]
        
        for question in test_questions:
            print(f"\n🤔 Pregunta: {question}")
            response = query_rag_system(question)
            print(f"Respuesta: {response['text']}")
            print(f"Imgenes: {len(response['images'])} imagen(es) seleccionada(s)")