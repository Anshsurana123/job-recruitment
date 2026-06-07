package com.example.salesvoice.ui.onboarding

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.view.LayoutInflater
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.LinearLayout
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.RecyclerView
import androidx.viewpager2.widget.ViewPager2
import com.example.salesvoice.BaseActivity
import com.example.salesvoice.MainActivity
import com.example.salesvoice.R
import com.example.salesvoice.databinding.ActivityOnboardingBinding
import com.example.salesvoice.databinding.ItemOnboardingSlideBinding

data class OnboardingSlide(
    val emoji: String,
    val title: String,
    val desc: String
)

class OnboardingActivity : BaseActivity() {

    private lateinit var binding: ActivityOnboardingBinding
    private lateinit var slides: List<OnboardingSlide>

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Check if first launch
        val sharedPrefs = getSharedPreferences("SalesVoicePrefs", Context.MODE_PRIVATE)
        val isFirstLaunch = sharedPrefs.getBoolean("is_first_launch", true)
        if (!isFirstLaunch) {
            launchMainActivity()
            return
        }

        binding = ActivityOnboardingBinding.inflate(layoutInflater)
        setContentView(binding.root)

        slides = listOf(
            OnboardingSlide(
                "🎙️",
                getString(R.string.onboarding_slide1_title),
                getString(R.string.onboarding_slide1_desc)
            ),
            OnboardingSlide(
                "📦",
                getString(R.string.onboarding_slide2_title),
                getString(R.string.onboarding_slide2_desc)
            ),
            OnboardingSlide(
                "🌐",
                getString(R.string.onboarding_slide3_title),
                getString(R.string.onboarding_slide3_desc)
            )
        )

        setupViewPager()
        setupIndicators()
        setCurrentIndicator(0)

        binding.btnNext.setOnClickListener {
            if (binding.viewPager.currentItem + 1 < slides.size) {
                binding.viewPager.currentItem += 1
            } else {
                // Save first launch flag
                sharedPrefs.edit().putBoolean("is_first_launch", false).apply()
                launchMainActivity()
            }
        }
    }

    private fun launchMainActivity() {
        startActivity(Intent(this, MainActivity::class.java))
        finish()
    }

    private fun setupViewPager() {
        val adapter = OnboardingAdapter(slides)
        binding.viewPager.adapter = adapter
        binding.viewPager.registerOnPageChangeCallback(object : ViewPager2.OnPageChangeCallback() {
            override fun onPageSelected(position: Int) {
                super.onPageSelected(position)
                setCurrentIndicator(position)
                if (position == slides.size - 1) {
                    binding.btnNext.text = getString(R.string.onboarding_get_started)
                } else {
                    binding.btnNext.text = getString(R.string.onboarding_next)
                }
            }
        })
    }

    private fun setupIndicators() {
        val indicators = arrayOfNulls<ImageView>(slides.size)
        val layoutParams = LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.WRAP_CONTENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ).apply {
            setMargins(8, 0, 8, 0)
        }

        for (i in indicators.indices) {
            indicators[i] = ImageView(applicationContext).apply {
                setImageDrawable(
                    ContextCompat.getDrawable(
                        applicationContext,
                        R.drawable.indicator_inactive
                    )
                )
                this.layoutParams = layoutParams
            }
            binding.llIndicators.addView(indicators[i])
        }
    }

    private fun setCurrentIndicator(index: Int) {
        val childCount = binding.llIndicators.childCount
        for (i in 0 until childCount) {
            val imageView = binding.llIndicators.getChildAt(i) as ImageView
            if (i == index) {
                imageView.setImageDrawable(
                    ContextCompat.getDrawable(
                        applicationContext,
                        R.drawable.indicator_active
                    )
                )
            } else {
                imageView.setImageDrawable(
                    ContextCompat.getDrawable(
                        applicationContext,
                        R.drawable.indicator_inactive
                    )
                )
            }
        }
    }

    inner class OnboardingAdapter(private val list: List<OnboardingSlide>) :
        RecyclerView.Adapter<OnboardingAdapter.SlideViewHolder>() {

        override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): SlideViewHolder {
            val binding = ItemOnboardingSlideBinding.inflate(
                LayoutInflater.from(parent.context), parent, false
            )
            return SlideViewHolder(binding)
        }

        override fun onBindViewHolder(holder: SlideViewHolder, position: Int) {
            holder.bind(list[position])
        }

        override fun getItemCount(): Int = list.size

        inner class SlideViewHolder(private val slideBinding: ItemOnboardingSlideBinding) :
            RecyclerView.ViewHolder(slideBinding.root) {

            fun bind(slide: OnboardingSlide) {
                slideBinding.tvSlideEmoji.text = slide.emoji
                slideBinding.tvSlideTitle.text = slide.title
                slideBinding.tvSlideDesc.text = slide.desc
            }
        }
    }
}
