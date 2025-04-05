from django.db import models

class Person(models.Model):
    nama = models.CharField(max_length=100)
    alamat = models.TextField()
    jabatan = models.CharField(max_length=100)
    gambar_0 = models.CharField(max_length=255, blank=True, null=True)  # Path untuk gambar pertama
    gambar_1 = models.CharField(max_length=255, blank=True, null=True)  # Path untuk gambar kedua
    gambar_2 = models.CharField(max_length=255, blank=True, null=True)  # Path untuk gambar ketiga

    def __str__(self):
        return self.nama

class Kehadiran(models.Model):
    namaAnggota = models.CharField(max_length=100)
    gambarKehadiran = models.ImageField(upload_to='kehadiran/')
    waktu = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.namaAnggota} - {self.waktu}"