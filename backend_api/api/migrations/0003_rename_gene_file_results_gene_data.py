# Generated by Django 4.1.3 on 2022-11-23 12:20

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0002_remove_results_path_to_data_results_gene_file'),
    ]

    operations = [
        migrations.RenameField(
            model_name='results',
            old_name='gene_file',
            new_name='gene_data',
        ),
    ]