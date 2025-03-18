 

import os
import cv2
import zipfile
import numpy as np
import faiss
import insightface
from django.shortcuts import render, redirect
from django.http import HttpResponse
from .forms import ImageSearchForm, GroupImageSearchForm
from .models import UploadedImage
from django.conf import settings
from django.core.files.storage import default_storage

# Load Face Recognition Model Once
model = insightface.app.FaceAnalysis(name='buffalo_l')
model.prepare(ctx_id=0, det_size=(640, 640))

# Function to extract embeddings from a single face image
def get_face_embedding(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    faces = model.get(img)
    return faces[0].embedding if faces else None

# Function to extract all face embeddings from a group image
def get_face_embeddings(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return []
    faces = model.get(img)
    return [face.embedding for face in faces]  # Return list of embeddings

# Home Page View
def home_view(request):
    return render(request, 'home.html')

# Single Image Search View
def single_image_search_view(request):
    matched_results = []
    if request.method == "POST":
        form = ImageSearchForm(request.POST, request.FILES)
        if form.is_valid():
            image_folder = form.cleaned_data['image_folder']
            query_image = form.cleaned_data['image']
            
            # Save query image
            uploaded_image = UploadedImage(image=query_image)
            uploaded_image.save()
            query_image_path = os.path.join(settings.MEDIA_ROOT, str(uploaded_image.image))

            # Check if folder exists
            if not os.path.exists(image_folder):
                return render(request, 'single_image_search.html', {"form": form, "error": "Folder path does not exist."})

            # Extract embeddings for folder images
            face_embeddings = {}
            for img_name in os.listdir(image_folder):
                img_path = os.path.join(image_folder, img_name)
                embedding = get_face_embedding(img_path)
                if embedding is not None:
                    face_embeddings[img_name] = embedding

            if not face_embeddings:
                return render(request, 'single_image_search.html', {"form": form, "error": "No face embeddings extracted!"})

            # Convert dictionary to NumPy array
            image_names = list(face_embeddings.keys())
            embeddings_matrix = np.array(list(face_embeddings.values()), dtype=np.float32)

            # Create FAISS index
            d = embeddings_matrix.shape[1]  # Dynamic dimension detection
            index = faiss.IndexFlatL2(d)
            index.add(embeddings_matrix)

            # Extract embedding from query image
            query_embedding = get_face_embedding(query_image_path)

            if query_embedding is not None:
                query_embedding = np.array([query_embedding], dtype=np.float32)
                D, I = index.search(query_embedding, k=min(6, len(image_names)))  # Get top-5 results

                # Display results
                for img_idx, dist in zip(I[0], D[0]):
                    if dist < 400:  # Apply threshold
                        matched_results.append({
                            "image_name": image_names[img_idx],
                            "distance": round(dist, 5)  # Rounded for better readability
                        })

                # Generate ZIP file
                zip_filename = generate_zip_file(image_folder, matched_results)

                return render(request, 'single_image_search.html', {
                    "form": form,
                    "matched_results": matched_results,
                    "zip_file": zip_filename
                })

            return render(request, 'single_image_search.html', {"form": form, "error": "No face detected in query image!"})

    else:
        form = ImageSearchForm()
    return render(request, 'single_image_search.html', {"form": form})

# Group Image Search View
def group_image_search_view(request):
    matched_results = []
    if request.method == "POST":
        form = GroupImageSearchForm(request.POST, request.FILES)
        if form.is_valid():
            image_folder = form.cleaned_data['image_folder']
            query_image = form.cleaned_data['image']
            
            # Save query image
            uploaded_image = UploadedImage(image=query_image)
            uploaded_image.save()
            query_image_path = os.path.join(settings.MEDIA_ROOT, str(uploaded_image.image))

            # Check if folder exists
            if not os.path.exists(image_folder):
                return render(request, 'group_image_search.html', {"form": form, "error": "Folder path does not exist."})

            # Process each image and store embeddings
            face_embeddings = {}
            for img_name in os.listdir(image_folder):
                img_path = os.path.join(image_folder, img_name)
                embeddings = get_face_embeddings(img_path)
                if embeddings:  
                    face_embeddings[img_name] = embeddings

            if not face_embeddings:
                return render(request, 'group_image_search.html', {"form": form, "error": "No face embeddings extracted!"})

            # Convert embeddings dictionary to FAISS-compatible format
            image_names = []
            embeddings_matrix = []
            image_face_map = {}  

            for img_name, embeddings in face_embeddings.items():
                for emb in embeddings:
                    image_names.append(img_name)
                    embeddings_matrix.append(emb)
                    image_face_map[len(image_names) - 1] = img_name   

            embeddings_matrix = np.array(embeddings_matrix, dtype=np.float32)

            # Initialize FAISS index (L2 Euclidean Distance)
            d = embeddings_matrix.shape[1]  
            index = faiss.IndexFlatL2(d)
            index.add(embeddings_matrix)

            # Extract embeddings from query image
            query_embeddings = get_face_embeddings(query_image_path)  

            if query_embeddings:
                for i, query_embedding in enumerate(query_embeddings):
                    query_embedding = np.array([query_embedding], dtype=np.float32)
                    D, I = index.search(query_embedding, k=min(5, len(image_names)))   

                    for rank, (img_idx, dist) in enumerate(zip(I[0], D[0]), start=1):
                        if dist < 400:  
                            matched_results.append({
                                "image_name": image_face_map[img_idx],
                                "distance": round(dist, 5)  
                            })

                # Generate ZIP file
                zip_filename = generate_zip_file(image_folder, matched_results)

                return render(request, 'group_image_search.html', {
                    "form": form,
                    "matched_results": matched_results,
                    "zip_file": zip_filename
                })

            return render(request, 'group_image_search.html', {"form": form, "error": "No face detected in query image!"})

    else:
        form = GroupImageSearchForm()
    return render(request, 'group_image_search.html', {"form": form})

# Generate ZIP file of matched images
def generate_zip_file(image_folder, matched_results):
    zip_path = os.path.join(settings.MEDIA_ROOT, "matched_images.zip")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for result in matched_results:
            img_path = os.path.join(image_folder, result["image_name"])
            if os.path.exists(img_path):
                zipf.write(img_path, arcname=result["image_name"])
    return zip_path

# Download ZIP file
def download_zip_file(request):
    zip_path = os.path.join(settings.MEDIA_ROOT, "matched_images.zip")
    if os.path.exists(zip_path):
        with open(zip_path, 'rb') as zipf:
            response = HttpResponse(zipf.read(), content_type='application/zip')
            response['Content-Disposition'] = 'attachment; filename="matched_images.zip"'
            return response
    return HttpResponse("File not found", status=404)