import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import Sidebar from '@/components/Sidebar';
import './globals.css';

const inter = Inter({ subsets: ['latin'], variable: '--font-inter' });

export const metadata: Metadata = {
  title: {
    default: 'Cerebro-X — Digital Brain Twin',
    template: '%s | Cerebro-X',
  },
  description:
    'An Explainable Multimodal AI system for Longitudinal Prediction of Alzheimer\'s Disease Progression using a Digital Brain Twin architecture.',
  keywords: ['Alzheimer', 'CDR prediction', 'digital brain twin', 'GRU', 'OASIS-2', 'explainability', 'SHAP'],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={inter.variable}>
      <body>
        <div className="app-shell">
          <Sidebar />
          <main className="main-content">
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}
