import base64
import cv2
import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from .models import Person, Kehadiran
import os
from django.conf import settings
from io import BytesIO
from PIL import Image
import json
import face_recognition
import pickle
from django.utils import timezone
import logging

# Setup logging
logging.basicConfig(filename='recognition.log', level=logging.INFO)

def capture_page(request):
    return render(request, 'capture.html')

def absen_page(request):
    return render(request, 'absen.html')

def attendance_list_page(request):
    attendances = Kehadiran.objects.all().order_by('-waktu')
    persons = Person.objects.all().order_by('nama')
    return render(request, 'attendance_list.html', {'attendances': attendances, 'persons': persons})

@csrf_exempt
def save_face(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            nama = data.get('nama')
            alamat = data.get('alamat')
            jabatan = data.get('jabatan')
            images = data.get('images')

            if not images or len(images) < 3:
                return JsonResponse({'status': 'error', 'message': '3 images required'}, status=400)

            person = Person(nama=nama, alamat=alamat, jabatan=jabatan)
            user_folder = os.path.join(settings.MEDIA_ROOT, 'faces', nama)
            os.makedirs(user_folder, exist_ok=True)

            for i, image_data in enumerate(images):
                if not image_data or not isinstance(image_data, str):
                    return JsonResponse({'status': 'error', 'message': f'Image {i+1} Salah Format'}, status=400)
                image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                image = Image.open(BytesIO(image_bytes))
                image_array = np.array(image)
                image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

                # face_locations = face_recognition.face_locations(image_rgb)
                # if len(face_locations) != 1:
                #     return JsonResponse({'status': 'error', 'message': f'Image {i+1}: Exactly one face required'}, status=400)
                
                face_locations = face_recognition.face_locations(image_rgb)

                # Jika tidak ada wajah
                if len(face_locations) == 0:
                    return JsonResponse({'status': 'error', 'message': f'Image {i+1}: No face detected'}, status=400)

                # Jika lebih dari satu wajah, pilih yang terbesar
                if len(face_locations) > 1:
                    # Hitung ukuran wajah dan pilih yang terbesar
                    face_sizes = [(bottom - top) * (right - left) for (top, right, bottom, left) in face_locations]
                    largest_face_idx = face_sizes.index(max(face_sizes))
                    face_locations = [face_locations[largest_face_idx]]  # Gunakan hanya wajah terbesar

                top, right, bottom, left = face_locations[0]
                face_image = image_array[top:bottom, left:right]
                cropped_image = Image.fromarray(face_image)

                file_name = f"{nama}_{i}.jpg"
                file_path = os.path.join(user_folder, file_name)
                cropped_image.save(file_path, 'JPEG')

                # Simpan path relatif ke database
                relative_path = f"faces/{nama}/{file_name}"
                if i == 0:
                    person.gambar_0 = relative_path
                elif i == 1:
                    person.gambar_1 = relative_path
                elif i == 2:
                    person.gambar_2 = relative_path
            person.save()  # Simpan perubahan ke database

            return JsonResponse({
                'status': 'success',
                'message': 'Data and face images saved successfully',
                'id': person.id
            })
        except Exception as e:
            logging.error(f"Save face error: {str(e)}")
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
    return JsonResponse({'status': 'error', 'message': 'Method not allowed'}, status=405)

@csrf_exempt
def train_faces_view(request):
    if request.method == "POST":
        base_image_dir = os.path.join(settings.MEDIA_ROOT, 'faces')
        known_encodings = []
        known_names = []

        for user_folder in os.listdir(base_image_dir):
            user_path = os.path.join(base_image_dir, user_folder)
            if os.path.isdir(user_path):
                for filename in os.listdir(user_path):
                    if filename.endswith(('.jpg', '.jpeg', '.png')):
                        image_path = os.path.join(user_path, filename)
                        image = face_recognition.load_image_file(image_path)
                        face_encodings = face_recognition.face_encodings(image)
                        if face_encodings:
                            encoding = face_encodings[0]
                            name = user_folder
                            known_encodings.append(encoding)
                            known_names.append(name)
                            logging.info(f"Trained: {name} from {image_path}")

        data = {"encodings": known_encodings, "names": known_names}
        with open(os.path.join(settings.MEDIA_ROOT, 'face_encodings.pkl'), 'wb') as f:
            pickle.dump(data, f)
        
        return JsonResponse({'status': 'success', 'message': 'Training completed'})
    return JsonResponse({'status': 'error', 'message': 'Method not allowed'}, status=405)

@csrf_exempt
def recognize_and_attend(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            images = data.get('images')  # List of frames

            if not images or not isinstance(images, list) or len(images) == 0:
                logging.error("No valid images received")
                return JsonResponse({'status': 'error', 'message': 'No valid images provided'}, status=400)

            with open(os.path.join(settings.MEDIA_ROOT, 'face_encodings.pkl'), 'rb') as f:
                trained_data = pickle.load(f)
            known_encodings = trained_data['encodings']
            known_names = trained_data['names']

            best_match = None
            best_distance = float('inf')
            best_image = None

            for image_data in images:
                if not image_data or not isinstance(image_data, str):
                    logging.warning(f"Invalid image data: {image_data}")
                    continue
                try:
                    image_data = image_data.split(',')[1]
                except (AttributeError, IndexError) as e:
                    logging.error(f"Image data split error: {str(e)}")
                    continue

                image_bytes = base64.b64decode(image_data)
                image = Image.open(BytesIO(image_bytes))
                image_array = np.array(image)
                image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                face_encodings = face_recognition.face_encodings(image_rgb)

                if len(face_encodings) == 0:
                    continue
                if len(face_encodings) > 1:
                    continue

                unknown_encoding = face_encodings[0]
                distances = face_recognition.face_distance(known_encodings, unknown_encoding)
                min_distance = min(distances)
                if min_distance < best_distance and min_distance < 0.5:
                    best_distance = min_distance
                    best_match = known_names[distances.argmin()]
                    face_locations = face_recognition.face_locations(image_rgb)
                    top, right, bottom, left = face_locations[0]
                    face_image = image_array[top:bottom, left:right]
                    best_image = Image.fromarray(face_image)

            if not best_match:
                return JsonResponse({'status': 'error', 'message': 'No match found'}, status=400)

            name = best_match
            today = timezone.now().date()
            if Kehadiran.objects.filter(namaAnggota=name, waktu__date=today).exists():
                return JsonResponse({'status': 'error', 'message': f'{name} already attended today'}, status=400)

            user_folder = os.path.join(settings.MEDIA_ROOT, 'kehadiran', name)
            os.makedirs(user_folder, exist_ok=True)
            file_name = f"{int(__import__('time').time())}.jpg"
            file_path = os.path.join(user_folder, file_name)
            best_image.save(file_path, 'JPEG')

            kehadiran = Kehadiran(gambarKehadiran=f'kehadiran/{name}/{file_name}', namaAnggota=name)
            kehadiran.save()

            logging.info(f"Recognized: {name}, Distance: {best_distance}")

            return JsonResponse({
                'status': 'success',
                'message': f'Face recognized as {name}, attendance recorded',
                'nama': name,
                'waktu': kehadiran.waktu.isoformat(),
                'gambar': f'{settings.MEDIA_URL}kehadiran/{name}/{file_name}'
            })
        except Exception as e:
            logging.error(f"Recognize and attend error: {str(e)}")
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
    return JsonResponse({'status': 'error', 'message': 'Method not allowed'}, status=405)

def get_today_attendances(request):
    today = timezone.now().date()
    attendances = Kehadiran.objects.filter(waktu__date=today).order_by('-waktu')
    data = [{'nama': att.namaAnggota, 'gambar': att.gambarKehadiran.url} for att in attendances]
    return JsonResponse({'attendances': data})

@csrf_exempt
def delete_attendance(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            attendance_id = data.get('id')
            attendance = Kehadiran.objects.get(id=attendance_id)
            image_path = os.path.join(settings.MEDIA_ROOT, attendance.gambarKehadiran.name)
            if os.path.exists(image_path):
                os.remove(image_path)
            attendance.delete()
            return JsonResponse({'status': 'success', 'message': 'Attendance deleted'})
        except Kehadiran.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': 'Attendance not found'}, status=404)
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
    return JsonResponse({'status': 'error', 'message': 'Method not allowed'}, status=405)

@csrf_exempt
def delete_person(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            person_id = data.get('id')
            person = Person.objects.get(id=person_id)
            user_folder = os.path.join(settings.MEDIA_ROOT, 'faces', person.nama)
            if os.path.exists(user_folder):
                for filename in os.listdir(user_folder):
                    os.remove(os.path.join(user_folder, filename))
                os.rmdir(user_folder)
            person.delete()
            train_faces_view(request)
            return JsonResponse({'status': 'success', 'message': 'Person deleted'})
        except Person.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': 'Person not found'}, status=404)
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
    return JsonResponse({'status': 'error', 'message': 'Method not allowed'}, status=405)

@csrf_exempt
def edit_person(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            person_id = data.get('id')
            nama = data.get('nama')
            alamat = data.get('alamat')
            jabatan = data.get('jabatan')
            person = Person.objects.get(id=person_id)

            old_folder = os.path.join(settings.MEDIA_ROOT, 'faces', person.nama)
            new_folder = os.path.join(settings.MEDIA_ROOT, 'faces', nama)
            if person.nama != nama and os.path.exists(old_folder):
                os.rename(old_folder, new_folder)

            person.nama = nama
            person.alamat = alamat
            person.jabatan = jabatan
            person.save()
            train_faces_view(request)
            return JsonResponse({'status': 'success', 'message': 'Person updated'})
        except Person.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': 'Person not found'}, status=404)
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
    return JsonResponse({'status': 'error', 'message': 'Method not allowed'}, status=405)