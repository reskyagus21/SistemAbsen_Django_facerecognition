# Generated by Django 5.0.6 on 2025-04-05 07:29

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('ApiAbsen', '0003_remove_person_created_at_remove_person_gambar_and_more'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='person',
            name='gambar_2',
        ),
    ]
