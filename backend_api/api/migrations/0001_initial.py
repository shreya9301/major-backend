# Generated by Django 4.1.3 on 2022-11-22 17:18

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='Results',
            fields=[
                ('resultid', models.AutoField(primary_key=True, serialize=False)),
                ('path_to_data', models.CharField(blank=None, max_length=100, unique=True)),
                ('prediction', models.CharField(blank=None, max_length=10)),
                ('username', models.ForeignKey(default=None, on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
    ]