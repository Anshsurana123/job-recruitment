package com.example.salesvoice.ui.products

import androidx.lifecycle.LiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.salesvoice.data.model.Product
import com.example.salesvoice.data.repository.ProductRepository
import kotlinx.coroutines.launch

class ProductsViewModel(private val productRepo: ProductRepository) : ViewModel() {

    val allProducts: LiveData<List<Product>> = productRepo.allProducts

    fun insertProduct(product: Product, onResult: (Long) -> Unit) {
        viewModelScope.launch {
            val id = productRepo.insert(product)
            onResult(id)
        }
    }

    fun updateProduct(product: Product) {
        viewModelScope.launch {
            productRepo.update(product)
        }
    }

    fun deleteProduct(product: Product) {
        viewModelScope.launch {
            productRepo.delete(product)
        }
    }

    suspend fun isDuplicateName(name: String, editingId: Long?): Boolean {
        val products = productRepo.getAllProductsSync()
        return products.any { 
            it.name.equals(name, ignoreCase = true) && (editingId == null || it.id != editingId) 
        }
    }
}
