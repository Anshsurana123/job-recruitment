package com.example.salesvoice.ui

import android.content.Context
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import com.example.salesvoice.data.db.AppDatabase
import com.example.salesvoice.data.repository.ProductRepository
import com.example.salesvoice.data.repository.SaleRepository
import com.example.salesvoice.ui.products.ProductsViewModel
import com.example.salesvoice.ui.record.RecordViewModel
import com.example.salesvoice.ui.report.ReportViewModel

class ViewModelFactory(private val context: Context) : ViewModelProvider.Factory {
    
    private val database by lazy { AppDatabase.getDatabase(context) }
    private val productRepository by lazy { ProductRepository(database.productDao()) }
    private val saleRepository by lazy { SaleRepository(database.saleDao()) }

    @Suppress("UNCHECKED_CAST")
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        return when {
            modelClass.isAssignableFrom(ProductsViewModel::class.java) -> {
                ProductsViewModel(productRepository) as T
            }
            modelClass.isAssignableFrom(RecordViewModel::class.java) -> {
                RecordViewModel(saleRepository, productRepository) as T
            }
            modelClass.isAssignableFrom(ReportViewModel::class.java) -> {
                ReportViewModel(saleRepository, productRepository) as T
            }
            else -> throw IllegalArgumentException("Unknown ViewModel class: ${modelClass.name}")
        }
    }
}
