import { create } from 'zustand';
import { Company } from '@/types/company';

interface CompanyState {
  companies: Company[];
  selectedCompany: Company | null;
  selectedYear: number | null;
  
  setCompanies: (companies: Company[]) => void;
  setSelectedCompany: (company: Company | null) => void;
  setSelectedYear: (year: number | null) => void;
}

export const useCompanyStore = create<CompanyState>((set) => ({
  companies: [],
  selectedCompany: null,
  selectedYear: null,

  setCompanies: (companies) => set({ companies }),
  setSelectedCompany: (company) => set({ selectedCompany: company }),
  setSelectedYear: (year) => set({ selectedYear: year }),
}));