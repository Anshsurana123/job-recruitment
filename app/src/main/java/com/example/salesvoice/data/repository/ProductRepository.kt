package com.example.salesvoice.data.repository

import androidx.lifecycle.LiveData
import com.example.salesvoice.data.db.ProductDao
import com.example.salesvoice.data.model.Product
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class ProductRepository(private val productDao: ProductDao) {

    val allProducts: LiveData<List<Product>> = productDao.getAllProducts()

    suspend fun getAllProductsSync(): List<Product> = withContext(Dispatchers.IO) {
        productDao.getAllProductsSync()
    }

    suspend fun insert(product: Product): Long = withContext(Dispatchers.IO) {
        productDao.insert(product)
    }

    suspend fun update(product: Product) = withContext(Dispatchers.IO) {
        productDao.update(product)
    }

    suspend fun delete(product: Product) = withContext(Dispatchers.IO) {
        productDao.delete(product)
    }
}
