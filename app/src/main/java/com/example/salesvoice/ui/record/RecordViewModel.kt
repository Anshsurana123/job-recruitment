package com.example.salesvoice.ui.record

import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.map
import androidx.lifecycle.viewModelScope
import com.example.salesvoice.data.model.DailyTotals
import com.example.salesvoice.data.model.SaleEntry
import com.example.salesvoice.data.repository.ProductRepository
import com.example.salesvoice.data.repository.SaleRepository
import com.example.salesvoice.parser.ProductMatcher
import com.example.salesvoice.parser.SpeechParser
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

sealed class SaleResult {
    data class Success(val entry: SaleEntry) : SaleResult()
    data class ProductNotFound(val productNameSpoken: String) : SaleResult()
    data class Error(val message: String) : SaleResult()
}

class RecordViewModel(
    private val saleRepo: SaleRepository,
    private val productRepo: ProductRepository
) : ViewModel() {

    val todaySales: LiveData<List<SaleEntry>> = saleRepo.getTodaySales()
    val todayTotals: LiveData<DailyTotals> = saleRepo.getTodayTotals()
    
    // LiveData to notify view of processed speech result
    private val _saleResult = MutableLiveData<SaleResult?>()
    val saleResult: LiveData<SaleResult?> get() = _saleResult

    private val _isProcessing = MutableLiveData<Boolean>(false)
    val isProcessing: LiveData<Boolean> get() = _isProcessing

    val allProductsCount: LiveData<Int> = productRepo.allProducts.map { it?.size ?: 0 }

    fun deleteSale(saleId: Long) {
        viewModelScope.launch {
            saleRepo.deleteSale(saleId)
        }
    }

    fun clearSaleResult() {
        _saleResult.value = null
    }

    fun processVoiceTranscript(transcript: String) {
        _isProcessing.value = true
        viewModelScope.launch {
            val parseResult = withContext(Dispatchers.Default) {
                SpeechParser.parse(transcript)
            }

            if (parseResult.error != null) {
                _saleResult.postValue(SaleResult.Error(parseResult.error))
                _isProcessing.postValue(false)
                return@launch
            }

            val qty = parseResult.quantity
            val productSpoken = parseResult.productNameSpoken

            if (qty == null || productSpoken == null) {
                _saleResult.postValue(SaleResult.Error("Couldn't extract quantity or product. Example: say '2 kg Rice'"))
                _isProcessing.postValue(false)
                return@launch
            }

            // Match product
            val products = productRepo.getAllProductsSync()
            val matchedProduct = ProductMatcher.findMatch(productSpoken, products)

            if (matchedProduct == null) {
                _saleResult.postValue(SaleResult.ProductNotFound(productSpoken))
                _isProcessing.postValue(false)
                return@launch
            }

            // Create and insert sale entry
            val totalAmount = qty * matchedProduct.pricePerUnit
            val totalProfit = qty * matchedProduct.profitPerUnit
            val saleEntry = SaleEntry(
                productId = matchedProduct.id,
                productName = matchedProduct.name,
                quantity = qty,
                unit = matchedProduct.unit,
                pricePerUnit = matchedProduct.pricePerUnit,
                profitPerUnit = matchedProduct.profitPerUnit,
                totalAmount = totalAmount,
                totalProfit = totalProfit,
                timestamp = System.currentTimeMillis()
            )

            saleRepo.insertSale(saleEntry)
            _saleResult.postValue(SaleResult.Success(saleEntry))
            _isProcessing.postValue(false)
        }
    }
}
