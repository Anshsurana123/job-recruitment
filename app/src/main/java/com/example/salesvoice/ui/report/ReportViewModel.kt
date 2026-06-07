package com.example.salesvoice.ui.report

import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.map
import androidx.lifecycle.switchMap
import com.example.salesvoice.data.model.DailyTotals
import com.example.salesvoice.data.model.SaleEntry
import com.example.salesvoice.data.repository.ProductRepository
import com.example.salesvoice.data.repository.SaleRepository
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Date
import java.util.Locale

data class ProductBreakdown(
    val productName: String,
    val totalQuantity: Double,
    val unit: String,
    val totalRevenue: Double,
    val totalProfit: Double
)

class ReportViewModel(
    private val saleRepo: SaleRepository,
    private val productRepo: ProductRepository
) : ViewModel() {

    private val calendar = Calendar.getInstance()

    private val _selectedDateStr = MutableLiveData<String>()
    val selectedDateStr: LiveData<String> get() = _selectedDateStr

    // Date object for formatting/visuals
    private val _currentDate = MutableLiveData<Date>()
    val currentDate: LiveData<Date> get() = _currentDate

    init {
        val today = Date()
        _currentDate.value = today
        _selectedDateStr.value = getFormattedDbDate(today)
    }

    private fun getFormattedDbDate(date: Date): String {
        val sdf = SimpleDateFormat("yyyy-MM-dd", Locale.US)
        return sdf.format(date)
    }

    fun moveToPreviousDay() {
        calendar.time = _currentDate.value ?: Date()
        calendar.add(Calendar.DAY_OF_YEAR, -1)
        val prevDate = calendar.time
        _currentDate.value = prevDate
        _selectedDateStr.value = getFormattedDbDate(prevDate)
    }

    fun moveToNextDay() {
        // Prevent moving past today
        val todayStr = getFormattedDbDate(Date())
        val currentStr = _selectedDateStr.value ?: todayStr
        if (currentStr == todayStr) return

        calendar.time = _currentDate.value ?: Date()
        calendar.add(Calendar.DAY_OF_YEAR, 1)
        val nextDate = calendar.time
        _currentDate.value = nextDate
        _selectedDateStr.value = getFormattedDbDate(nextDate)
    }

    val salesForDate: LiveData<List<SaleEntry>> = _selectedDateStr.switchMap { dateStr ->
        saleRepo.getSalesForDate(dateStr)
    }

    val totalsForDate: LiveData<DailyTotals> = _selectedDateStr.switchMap { dateStr ->
        saleRepo.getTotalsForDate(dateStr)
    }

    val productBreakdowns: LiveData<List<ProductBreakdown>> = salesForDate.map { sales ->
        sales.groupBy { it.productId }.map { (_, entryList) ->
            val firstEntry = entryList.first()
            ProductBreakdown(
                productName = firstEntry.productName,
                totalQuantity = entryList.sumOf { it.quantity },
                unit = firstEntry.unit,
                totalRevenue = entryList.sumOf { it.totalAmount },
                totalProfit = entryList.sumOf { it.totalProfit }
            )
        }.sortedBy { it.productName }
    }
}
