import apiClient from './client';

export interface UploadProgress {
  status: 'uploading' | 'processing' | 'success' | 'error';
  progress: number;
  message: string;
}

export const uploadApi = {
  uploadReport: async (
    file: File,
    ticker: string,
    companyName: string,
    year: number,
    onProgress?: (progress: UploadProgress) => void
  ) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('ticker', ticker);
    formData.append('company_name', companyName);
    formData.append('year', year.toString());

    const response = await apiClient.post('/api/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          const percentCompleted = Math.round(
            (progressEvent.loaded * 100) / progressEvent.total
          );
          onProgress({
            status: 'uploading',
            progress: percentCompleted,
            message: `Uploading: ${percentCompleted}%`,
          });
        }
      },
    });

    return response.data;
  },
};