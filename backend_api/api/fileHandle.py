import os
import datetime

def create_dirs(username):
        Base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        Dir = os.path.join(Base_dir,"media/")
        final_dir = os.path.join(Dir,str(username)+"/")
        print(final_dir)
        print(os.path.isdir(final_dir))
        if (os.path.isdir(final_dir))== False :
                os.mkdir(final_dir)
        
        #final_dir = os.path.join(final_dir,str(datetime.datetime.now().date()))
        
        if (os.path.isdir(final_dir))== False :
                os.mkdir(final_dir)

        return final_dir


def handle_uploaded_file(username,gene_data):
    final_path = create_dirs(username) 
    final_path = os.path.join(final_path,str(gene_data))
    return final_path
